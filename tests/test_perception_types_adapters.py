# ruff: noqa: F403, F405
from tests.perception_helpers import *  # noqa: F403


def test_artifact_rejects_invalid_dimensions_and_scheme() -> None:
    with pytest.raises(ValueError):
        ArtifactRef("x", "file:///x", "image/jpeg", (0, 1), "uint8", 1)
    with pytest.raises(ValueError):
        ArtifactRef("x", "http://x", "image/jpeg", (1, 1), "uint8", 1)


def test_perception_input_rejects_duplicate_cameras() -> None:
    camera = artifact("camera_front")
    with pytest.raises(ValueError):
        PerceptionInput(1, 0.1, (camera, camera), None, artifact("cal", "application/json"))


def test_adapter_rejects_missing_lidar_and_sensor_skew() -> None:
    adapter = BevFusionDetectionAdapter(
        adapter_requirements(CAMERAS, True), "mit-bevfusion-swint"
    )
    with pytest.raises(PerceptionInputError):
        adapter.validate_input(input_value(with_lidar=False))
    stale = input_value(timestamp=2.0)
    stale_camera = artifact("camera_front", timestamp=1.0)
    stale = PerceptionInput(
        stale.frame,
        stale.timestamp,
        (stale_camera,) + stale.cameras[1:],
        stale.lidar,
        stale.calibration,
    )
    with pytest.raises(PerceptionInputError):
        adapter.validate_input(stale)


def test_bevfusion_normalizes_3d_detection() -> None:
    adapter = BevFusionDetectionAdapter(
        adapter_requirements(CAMERAS, True), "mit-bevfusion-swint-v0p075-convfuser"
    )
    output = adapter.parse_response(input_value(), detection_payload())
    assert output.kind is PerceptionOutputKind.OBJECT_DETECTION_3D
    assert output.detections_3d[0].class_name == "car"
    assert output.diagnostics["inference_ms"] == 20.0


def test_bevfusion_segmentation_normalizes_single_bev_raster() -> None:
    adapter = BevFusionSegmentationAdapter(
        adapter_requirements(CAMERAS, True), "mit-bevfusion-fusion-bev256d2-lss"
    )
    output = adapter.parse_response(input_value(), raster_payload())
    assert output.kind is PerceptionOutputKind.BEV_SEGMENTATION
    assert len(output.rasters) == 1


def test_maptr_normalizes_vector_map() -> None:
    adapter = MapTRv2Adapter(
        adapter_requirements(CAMERAS, False), "maptrv2-r50-bevpool-24ep"
    )
    output = adapter.parse_response(
        input_value(),
        {
            "vector_map": [
                {
                    "category": "lane_divider",
                    "confidence": 0.8,
                    "points_xyz_m": [[0, 0, 0], [5, 0, 0]],
                }
            ]
        },
    )
    assert output.kind is PerceptionOutputKind.VECTOR_MAP
    assert len(output.vector_map[0].points_xyz_m) == 2


def test_tld_ready_normalizes_state_and_relevance() -> None:
    adapter = TldReadyAdapter(
        adapter_requirements(("camera_front",), False), "tld-ready-yolov8x"
    )
    output = adapter.parse_response(
        input_value(),
        {
            "traffic_lights": [
                {
                    "camera_name": "camera_front",
                    "bbox_xyxy": [1, 2, 5, 8],
                    "state": "red",
                    "pictogram": "circle",
                    "confidence": 0.95,
                    "relevant_to_ego": True,
                }
            ]
        },
    )
    assert output.traffic_lights[0].state == "RED"
    assert output.traffic_lights[0].relevant_to_ego is True


def test_citysemsegformer_supports_multiple_camera_masks() -> None:
    adapter = CitySemSegFormerAdapter(
        adapter_requirements(("camera_front", "camera_front_left"), False),
        "nvidia-citysemsegformer-deployable-onnx",
    )
    first = raster_payload("camera_front_semantic")["rasters"][0]
    second = raster_payload("camera_front_left_semantic")["rasters"][0]
    output = adapter.parse_response(input_value(), {"rasters": [first, second]})
    assert len(output.rasters) == 2
