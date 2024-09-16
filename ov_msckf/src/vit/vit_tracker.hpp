// Copyright 2024, Technical University of Munich.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  TODO@mateosss
 * @author Mateo de Mayo <mateo.demayo@tum.de>
 *
 * Similar format to https://gitlab.freedesktop.org/mateosss/basalt/-/blob/main/include/basalt/vit/vit_tracker.hpp
 */

#pragma once

#include "vit_implementation_helper.hpp"
#include <memory>

struct OVTracker final : vit::Tracker {
	OVTracker(const vit::Config *config);
	~OVTracker() override = default;

	vit::Result has_image_format(vit::ImageFormat fmt, bool *out_supported) const override;
	vit::Result get_supported_extensions(vit::TrackerExtensionSet *out_exts) const override;
	vit::Result get_enabled_extensions(vit::TrackerExtensionSet *out_exts) const override;
	vit::Result enable_extension(vit::TrackerExtension extension, bool enabled) override;
	vit::Result start() override;
	vit::Result stop() override;
	vit::Result reset() override;
	vit::Result is_running(bool *running) const override;
	vit::Result add_imu_calibration(const vit::ImuCalibration *calibration) override;
	vit::Result add_camera_calibration(const vit::CameraCalibration *calibration) override;
	vit::Result push_imu_sample(const vit::ImuSample *sample) override;
	vit::Result push_img_sample(const vit::ImgSample *sample) override;
	vit::Result pop_pose(vit::Pose **pose) override;
	vit::Result get_timing_titles(vit::TrackerTimingTitles *out_titles) const override;

  private:
	struct Implementation;
	std::unique_ptr<Implementation> impl_;
};

struct OVPose final : vit::Pose {
	~OVPose() override = default;

	vit::Result get_data(vit::PoseData *out_data) const override;
	vit::Result get_timing(vit::PoseTiming *out_timing) const override;
	vit::Result get_features(uint32_t camera_index, vit::PoseFeatures *out_features) const override;

  private:
	friend OVTracker;
	struct Implementation;
	std::unique_ptr<Implementation> impl_;
};
