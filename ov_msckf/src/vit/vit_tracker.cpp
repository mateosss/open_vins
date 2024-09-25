// Copyright 2024, Technical University of Munich.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Expose OpenVINS functionality through the Monado VIT API
 * @author Mateo de Mayo <mateo.demayo@tum.de>
 */

#include "vit_tracker.hpp"
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "moodycamel/blockingconcurrentqueue.h"
#include "moodycamel/concurrentqueue.h"
#include "moodycamel/readerwriterqueue.h"
#include "state/State.h"
#include "utils/print.h"
#include "utils/sensor_data.h"
#include "vit_implementation_helper.hpp"
#include "vit_interface.h"
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <utility>

constexpr size_t IMU_QUEUE_INITIAL_SIZE = 128;
constexpr size_t FRAME_QUEUE_INITIAL_SIZE = 16;
constexpr size_t POSE_QUEUE_INITIAL_SIZE = 128;

using namespace vit;
using namespace ov_msckf;
using namespace ov_core;

using moodycamel::BlockingConcurrentQueue;
using moodycamel::ConcurrentQueue;
using moodycamel::ReaderWriterQueue;
using std::atomic;
using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::string;
using std::thread;
using std::vector;
using StampedFrame = std::pair<double, cv::Mat>;

constexpr double S_AS_NS = 1e-9;

#define ASSERT(cond, ...)                                                                                              \
	do {                                                                                                               \
		if (!(cond)) {                                                                                                 \
			printf(MAGENTA "Assertion failed @%s:%d\n", __func__, __LINE__);                                           \
			printf(__VA_ARGS__);                                                                                       \
			printf("\n" RESET);                                                                                        \
			exit(EXIT_FAILURE);                                                                                        \
		}                                                                                                              \
	} while (false);
#define ASSERT_(cond) ASSERT(cond, "%s", #cond);

struct OVPose::Implementation {
	PoseData data{};
	Implementation(const State &state) {
		auto pos = state._imu->pos();
		auto quat = state._imu->quat();
		auto vel = state._imu->vel();
		data.timestamp = int64_t(state._timestamp / S_AS_NS);
		data.px = float(pos.x());
		data.py = float(pos.y());
		data.pz = float(pos.z());
		data.ox = float(quat.x());
		data.oy = float(quat.y());
		data.oz = float(quat.z());
		data.ow = float(quat.w());
		data.vx = float(vel.x());
		data.vy = float(vel.y());
		data.vz = float(vel.z());
	}
	~Implementation() = default;

	Result get_data(PoseData *out_data) const {
		*out_data = data;
		return VIT_SUCCESS;
	}

	Result get_timing(PoseTiming * /* out_timing */) const { return VIT_ERROR_NOT_SUPPORTED; }

	Result get_features(uint32_t /* cam_id */, PoseFeatures * /* out_feat */) const { return VIT_ERROR_NOT_SUPPORTED; }
};

Result OVPose::get_data(PoseData *data) const {
	impl_->get_data(data);
	return VIT_SUCCESS;
}

Result OVPose::get_timing(PoseTiming *out_timing) const { return impl_->get_timing(out_timing); }

Result OVPose::get_features(uint32_t camera_index, PoseFeatures *out_features) const {
	return impl_->get_features(camera_index, out_features);
}

struct OVTracker::Implementation {
	Implementation(const Config *config)
		: config_file(config->file), cam_count(config->cam_count), imu_count(config->imu_count),
		  show_ui(config->show_ui) {
		load_config();
		unsent_frames = vector<BlockingConcurrentQueue<StampedFrame>>(cam_count);
		ASSERT(imu_count == 1, "Unsupported imu_cout=%u", imu_count);
	}

	void load_config() {
		auto parser = make_shared<YamlParser>(config_file);

		string verbosity = "DEBUG"; // Default verbosity
		parser->parse_config("verbosity", verbosity);
		Printer::setPrintLevel(verbosity);

		params.print_and_load(parser);

		ASSERT(parser->successful(), "Unable to parse all parameters");
	}

	Result has_image_format(ImageFormat fmt, bool *out) const {
		if (fmt == VIT_IMAGE_FORMAT_L8) {
			*out = true;
			return VIT_SUCCESS;
		} else {
			PRINT_WARNING("Invalid image format: %d\n" RESET, fmt);
			return VIT_ERROR_INVALID_VALUE;
		}
	}

	Result get_supported_extensions(TrackerExtensionSet *out_exts) const {
		*out_exts = exts;
		return VIT_SUCCESS;
	}

	Result get_enabled_extensions(TrackerExtensionSet *out_exts) const {
		*out_exts = enabled_exts;
		return VIT_SUCCESS;
	}

	Result enable_extension(TrackerExtension ext, bool enable) {
		if (ext >= VIT_TRACKER_EXTENSION_COUNT) {
			PRINT_ERROR("Invalid extension: %d", ext);
			return VIT_ERROR_INVALID_VALUE;
		}

		bool supported = exts.has[ext];
		if (!supported) {
			PRINT_ERROR("Unsupported extension: %d", ext);
			return VIT_ERROR_NOT_SUPPORTED;
		}

		enabled_exts.has[ext] = enable;

		return VIT_SUCCESS;
	}

	Result initialize() { return VIT_SUCCESS; }

	Result start() {
		running = true;
		sys = make_shared<VioManager>(params);
		frame_consumer_thread = thread(&Implementation::frame_consumer, this);
		return VIT_SUCCESS;
	}

	vector<ImuData> get_unsent_imus_upto(double ts) {
		size_t step = IMU_QUEUE_INITIAL_SIZE;

		vector<ImuSample> imus_vit{};
		imus_vit.reserve(unsent_imus.size_approx());

		// Flush unsent_imus queue into imus_vit
		size_t dequeued = UINT64_MAX;
		size_t total_dequeued = 0;
		do {
			imus_vit.resize(imus_vit.size() + step); // Expand
			dequeued = unsent_imus.try_dequeue_bulk(imus_vit.begin() + int(total_dequeued), step);
			total_dequeued += dequeued;
		} while (dequeued >= step);
		imus_vit.resize(total_dequeued); // Shrink

		// Convert samples
		for (size_t i = 0; i < dequeued; i++) {
			const ImuSample &imu_vit = imus_vit[i];
			ImuData imu_ov{};
			imu_ov.timestamp = (double)imu_vit.timestamp * S_AS_NS;
			imu_ov.am << imu_vit.ax, imu_vit.ay, imu_vit.az;
			imu_ov.wm << imu_vit.wx, imu_vit.wy, imu_vit.wz;
			imus_ov.push_back(imu_ov);
		}

		// Find index i just after ts, assumes dequeued_imus is ordered
		int i = 0;
		for (; i < int(imus_ov.size()); i++)
			if (imus_ov[i].timestamp > ts) break;

		// Erase and return samples <=ts, keep the others for later
		vector<ImuData> unsent_imus_upto_ts{imus_ov.begin(), imus_ov.begin() + i};
		imus_ov.erase(imus_ov.begin(), imus_ov.begin() + i);
		return unsent_imus_upto_ts;
	}

	void frame_consumer() {
		const auto timeout = std::chrono::milliseconds(200);
		vector<bool> dequeued(cam_count);

		while (running) {
			CameraData frameset{};
			frameset.images.reserve(cam_count);
			frameset.sensor_ids.reserve(cam_count);
			frameset.masks.reserve(cam_count);
			for (int i = 0; i < int(cam_count); i++) {
				StampedFrame ts_img{};
				bool dequeued = unsent_frames.at(i).wait_dequeue_timed(ts_img, timeout);
				if (!dequeued) {
					if (i == 0) break;
					else ASSERT(i == 0, "Failed to dequeue cam%d frame", i);
				}

				auto [ts, img] = ts_img;
				if (i == 0) frameset.timestamp = ts;
				else if (ts != frameset.timestamp) break; // Ignore incomplete frameset

				frameset.images.push_back(img);
				frameset.sensor_ids.push_back(i);

				// TODO@mateosss: I am creating a new empty mask for each frame
				frameset.masks.push_back(cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8UC1));
			}

			if (frameset.images.size() != cam_count) {
				if (!frameset.images.empty())
					printf("Skipping %lu frames in %lf\n", frameset.images.size(), frameset.timestamp);
				continue;
			}

			auto imus = get_unsent_imus_upto(frameset.timestamp);
			for (const auto &measure_imu : imus) sys->feed_measurement_imu(measure_imu);

			sys->feed_measurement_camera(frameset);

#if 1					   // TODO@mateosss: delete?
			if (show_ui) { // display tracking result
				cv::Mat tracked_img = sys->get_historical_viz_image();
				if (!tracked_img.empty()) { cv::imshow("tracked_img", tracked_img); }
				int key = cv::waitKey(4);
				if (key == 'q' || key == 'Q') { break; }
			}
#endif

			if (!sys->initialized()) continue;

			shared_ptr<State> state_ov = sys->get_state();
			OVPose *p = new OVPose();
			p->impl_ = make_unique<OVPose::Implementation>(*state_ov);
			estimates.enqueue(p);
		}
	}

	Result finalize() { return VIT_SUCCESS; }

	Result stop() {
		running = false;
		frame_consumer_thread.join();
		return VIT_SUCCESS;
	}

	Result reset() { return VIT_ERROR_NOT_SUPPORTED; };

	bool is_running() { return running; };

	Result push_imu_sample(const ImuSample *imu_vit) {
		unsent_imus.enqueue(*imu_vit);
		return VIT_SUCCESS;
	}

	Result push_img_sample(const ImgSample *img_vit) {
		ASSERT(img_vit->format == VIT_IMAGE_FORMAT_L8, "Image format is not L8: %d", img_vit->format);

		uint32_t i = img_vit->cam_index;
		double ts = (double)img_vit->timestamp * S_AS_NS;
		int h = (int)img_vit->height;
		int w = (int)img_vit->width;
		int fmt = CV_8UC1;
		size_t stride = img_vit->stride;
		void *data_ptr = img_vit->data;
		cv::Mat img_wrapper{h, w, fmt, data_ptr, stride};
		cv::Mat img_cv = img_wrapper.clone();
		unsent_frames.at(i).enqueue({ts, img_cv});

		return VIT_SUCCESS;
	}

	Result pop_pose(Pose **out_pose) {
		OVPose *state{};

		bool popped = estimates.try_dequeue(state);

		if (popped) {
			if (out_pose == nullptr) return VIT_SUCCESS;
			*out_pose = state;
		} else {
			*out_pose = nullptr;
		}

		return vit::Result::VIT_SUCCESS;
	}

	Result get_timing_titles(TrackerTimingTitles * /*out_titles*/) const { return VIT_ERROR_NOT_SUPPORTED; }
	Result add_imu_calibration(const ImuCalibration * /*calibration*/) { return VIT_ERROR_NOT_SUPPORTED; }
	Result add_camera_calibration(const CameraCalibration * /*calibration*/) { return VIT_ERROR_NOT_SUPPORTED; }

  private:
	static constexpr TrackerExtensionSet exts{}; // No extensions
	TrackerExtensionSet enabled_exts{};
	string config_file;
	uint32_t cam_count = 0;
	uint32_t imu_count = 0;
	bool show_ui = false;
	VioManagerOptions params{};
	shared_ptr<VioManager> sys{};
	atomic<bool> running = false;
	std::vector<BlockingConcurrentQueue<StampedFrame>> unsent_frames{FRAME_QUEUE_INITIAL_SIZE};
	ConcurrentQueue<ImuSample> unsent_imus{IMU_QUEUE_INITIAL_SIZE};
	ReaderWriterQueue<OVPose *> estimates{POSE_QUEUE_INITIAL_SIZE};
	vector<ImuData> imus_ov;
	thread frame_consumer_thread{};
};

OVTracker::OVTracker(const Config *config) { impl_ = make_unique<OVTracker::Implementation>(config); }
Result OVTracker::has_image_format(ImageFormat fmt, bool *out) const { return impl_->has_image_format(fmt, out); }
Result OVTracker::get_supported_extensions(TrackerExtensionSet *out) const {
	return impl_->get_supported_extensions(out);
}
Result OVTracker::get_enabled_extensions(TrackerExtensionSet *out) const { return impl_->get_enabled_extensions(out); }
Result OVTracker::enable_extension(TrackerExtension ext, bool enable) { return impl_->enable_extension(ext, enable); }

Result OVTracker::start() {
	Result res = VIT_SUCCESS;

	res = impl_->initialize();
	if (res != VIT_SUCCESS) return res;

	res = impl_->start();
	if (res != VIT_SUCCESS) return res;

	return res;
}

Result OVTracker::stop() {
	Result res = VIT_SUCCESS;

	res = impl_->finalize();
	if (res != VIT_SUCCESS) return res;

	res = impl_->stop();
	if (res != VIT_SUCCESS) return res;

	return VIT_SUCCESS;
}

Result OVTracker::reset() { return impl_->reset(); }

Result OVTracker::is_running(bool *out_running) const {
	*out_running = impl_->is_running();
	return VIT_SUCCESS;
}

Result OVTracker::push_imu_sample(const ImuSample *sample) { return impl_->push_imu_sample(sample); }

Result OVTracker::push_img_sample(const ImgSample *sample) { return impl_->push_img_sample(sample); }

Result OVTracker::pop_pose(Pose **pose) { return impl_->pop_pose(pose); }

Result OVTracker::get_timing_titles(vit_tracker_timing_titles *out_titles) const {
	return impl_->get_timing_titles(out_titles);
}

Result OVTracker::add_imu_calibration(const ImuCalibration *calibration) {
	return impl_->add_imu_calibration(calibration);
}

Result OVTracker::add_camera_calibration(const CameraCalibration *calibration) {
	return impl_->add_camera_calibration(calibration);
}

Result vit_tracker_create(const Config *config, Tracker **out_tracker) {
	try {
		Tracker *tracker = new OVTracker(config);
		*out_tracker = tracker;
	} catch (const std::exception &exc) {
		std::cerr << exc.what() << std::endl;
		return VIT_ERROR_ALLOCATION_FAILURE;
	}

	return VIT_SUCCESS;
}
