import numpy as np
import pytest

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    Batch,
    BatchProvider,
    BatchRequest,
    Coordinate,
    MergeProvider,
    RandomLocation,
    Roi,
    build,
    compute_mask_integral,
)
from gunpowder.pipeline import PipelineRequestError


class ExampleSourceRandomLocation(BatchProvider):
    def __init__(self, array):
        self.array = array
        self.roi = Roi((-200, -20, -20), (1000, 100, 100))
        self.data_shape = (60, 60, 60)
        self.voxel_size = Coordinate(20, 2, 2)
        x = np.linspace(-10, 49, 60).reshape((-1, 1, 1))
        self.data = x + x.transpose([1, 2, 0]) + x.transpose([2, 0, 1])

    def setup(self):
        self.provides(self.array, ArraySpec(roi=self.roi, voxel_size=self.voxel_size))

    def provide(self, request):
        batch = Batch()

        spec = request[self.array].copy()
        spec.voxel_size = self.voxel_size

        start = (request[self.array].roi.begin / self.voxel_size) + 10
        end = (request[self.array].roi.end / self.voxel_size) + 10
        data_slices = tuple(map(slice, start, end))

        data = self.data[data_slices]

        batch.arrays[self.array] = Array(data=data, spec=spec)

        return batch


class CustomRandomLocation(RandomLocation):
    def __init__(self, array, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.array = array

    # only accept random locations that contain (0, 0, 0)
    def accepts(self, request):
        return request.array_specs[self.array].roi.contains((0, 0, 0))


def test_random_shift():
    a = ArrayKey("A")
    b = ArrayKey("B")
    random_shift_key = ArrayKey("RANDOM_SHIFT")

    pipeline = (
        (ExampleSourceRandomLocation(a), ExampleSourceRandomLocation(b))
        + MergeProvider()
        + CustomRandomLocation(a, random_shift_key=random_shift_key)
    )
    pipeline_no_random = (
        ExampleSourceRandomLocation(a),
        ExampleSourceRandomLocation(b),
    ) + MergeProvider()

    with build(pipeline), build(pipeline_no_random):
        sums = set()
        for i in range(10):
            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        a: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20))),
                        b: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20))),
                        random_shift_key: ArraySpec(nonspatial=True),
                    }
                )
            )

            assert 0 in batch.arrays[a].data
            assert 0 in batch.arrays[b].data

            # check that we can repeat this request without the random location
            batch_no_random = pipeline_no_random.request_batch(
                BatchRequest(
                    {
                        a: ArraySpec(
                            roi=Roi(batch[random_shift_key].data, (20, 20, 20))
                        ),
                        b: ArraySpec(
                            roi=Roi(batch[random_shift_key].data, (20, 20, 20))
                        ),
                    }
                )
            )

            assert batch_no_random.arrays[a].data.sum() == batch.arrays[a].data.sum()

            sums.add(batch[a].data.sum())

            # Request a ROI with the same shape as the entire ROI
            full_roi_a = Roi((0, 0, 0), ExampleSourceRandomLocation(a).roi.shape)
            full_roi_b = Roi((0, 0, 0), ExampleSourceRandomLocation(b).roi.shape)
            batch = pipeline.request_batch(
                BatchRequest(
                    {a: ArraySpec(roi=full_roi_a), b: ArraySpec(roi=full_roi_b)}
                )
            )
        assert len(sums) > 1


def test_random_location():
    a = ArrayKey("A")
    b = ArrayKey("B")
    source_a = ExampleSourceRandomLocation(a)
    source_b = ExampleSourceRandomLocation(b)

    pipeline = (source_a, source_b) + MergeProvider() + CustomRandomLocation(a)

    with build(pipeline):
        for i in range(10):
            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        a: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20))),
                        b: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20))),
                    }
                )
            )

            assert 0 in batch.arrays[a].data
            assert 0 in batch.arrays[b].data

            # Request a ROI with the same shape as the entire ROI
            full_roi_a = Roi((0, 0, 0), source_a.roi.shape)
            full_roi_b = Roi((0, 0, 0), source_b.roi.shape)
            batch = pipeline.request_batch(
                BatchRequest(
                    {a: ArraySpec(roi=full_roi_a), b: ArraySpec(roi=full_roi_b)}
                )
            )


def test_random_seed():
    raw = ArrayKey("RAW")
    pipeline = ExampleSourceRandomLocation(raw) + CustomRandomLocation(raw)

    with build(pipeline):
        seeded_sums = []
        unseeded_sums = []
        for i in range(10):
            batch_seeded = pipeline.request_batch(
                BatchRequest(
                    {raw: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20)))},
                    random_seed=10,
                )
            )
            seeded_sums.append(batch_seeded[raw].data.sum())
            batch_unseeded = pipeline.request_batch(
                BatchRequest({raw: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20)))})
            )
            unseeded_sums.append(batch_unseeded[raw].data.sum())

        assert len(set(seeded_sums)) == 1
        assert len(set(unseeded_sums)) > 1


def test_impossible():
    a = ArrayKey("A")
    b = ArrayKey("B")
    null_key = ArrayKey("NULL")
    source_a = ExampleSourceRandomLocation(a)
    source_b = ExampleSourceRandomLocation(b)

    pipeline = (source_a, source_b) + MergeProvider() + CustomRandomLocation(null_key)

    with build(pipeline):
        with pytest.raises(PipelineRequestError):
            pipeline.request_batch(
                BatchRequest(
                    {
                        a: ArraySpec(roi=Roi((0, 0, 0), (200, 20, 20))),
                        b: ArraySpec(roi=Roi((1000, 100, 100), (220, 22, 22))),
                    }
                )
            )


class MaskSourceRandomLocation(BatchProvider):
    """Provides a binary mask with a masked-in cube, for min_masked tests."""

    def __init__(self, key, shape=(40, 40, 40), voxel_size=(1, 1, 1)):
        self.key = key
        self.voxel_size = Coordinate(voxel_size)
        self.roi = Roi((0, 0, 0), Coordinate(shape) * self.voxel_size)
        self.data = np.zeros(shape, dtype=np.uint8)
        self.data[8:32, 8:32, 8:32] = 1  # masked-in cube

    def setup(self):
        self.provides(
            self.key,
            ArraySpec(roi=self.roi, voxel_size=self.voxel_size, interpolatable=False),
        )

    def provide(self, request):
        batch = Batch()
        roi = request[self.key].roi
        start = roi.begin / self.voxel_size
        end = roi.end / self.voxel_size
        data_slices = tuple(map(slice, start, end))
        spec = self.spec[self.key].copy()
        spec.roi = roi
        batch[self.key] = Array(self.data[data_slices].copy(), spec)
        return batch


def test_precomputed_mask_integral_matches_internal():
    # the integral RandomLocation builds internally must equal compute_mask_integral
    m = ArrayKey("MASK")
    src = MaskSourceRandomLocation(m)
    rl = RandomLocation(min_masked=0.5, mask=m)
    with build(src + rl):
        internal = np.asarray(rl.mask_integral)
    expected = compute_mask_integral(src.data)
    np.testing.assert_array_equal(internal, expected)
    assert internal.dtype == expected.dtype


def _run_masked_locations(rl, m, n=20):
    pipeline = MaskSourceRandomLocation(m) + rl
    sums = []
    with build(pipeline):
        for i in range(n):
            batch = pipeline.request_batch(
                BatchRequest(
                    {m: ArraySpec(roi=Roi((0, 0, 0), (8, 8, 8)))},
                    random_seed=100 + i,
                )
            )
            sums.append(int(batch[m].data.sum()))
    return sums


def test_precomputed_mask_integral_array_identical_behavior():
    # passing a precomputed integral array must give bit-identical placements to
    # letting RandomLocation compute it (the integral is a pure function of mask)
    m = ArrayKey("MASK")
    precomputed = compute_mask_integral(MaskSourceRandomLocation(m).data)
    stock = _run_masked_locations(RandomLocation(min_masked=0.5, mask=m), m)
    pre = _run_masked_locations(
        RandomLocation(min_masked=0.5, mask=m, mask_integral=precomputed), m
    )
    assert stock == pre
    assert min_masked_ok(pre)


def min_masked_ok(sums):
    # min_masked=0.5 over an 8^3 window -> at least half masked -> nonzero sum
    return all(s > 0 for s in sums)


def test_precomputed_mask_integral_from_npy(tmp_path):
    # a .npy path is memory-mapped read-only and behaves identically
    m = ArrayKey("MASK")
    precomputed = compute_mask_integral(MaskSourceRandomLocation(m).data)
    path = str(tmp_path / "mask_integral.npy")
    np.save(path, precomputed)

    rl = RandomLocation(min_masked=0.5, mask=m, mask_integral=path)
    src = MaskSourceRandomLocation(m)
    with build(src + rl):
        assert isinstance(rl.mask_integral, np.memmap)  # shared via page cache
        np.testing.assert_array_equal(np.asarray(rl.mask_integral), precomputed)

    stock = _run_masked_locations(RandomLocation(min_masked=0.5, mask=m), m)
    mmapped = _run_masked_locations(
        RandomLocation(min_masked=0.5, mask=m, mask_integral=path), m
    )
    assert stock == mmapped


def test_precomputed_mask_integral_zarr_array():
    # a zarr array (lazy, chunked) is read directly by integrate -> identical
    zarr = pytest.importorskip("zarr")
    m = ArrayKey("MASK")
    integ = compute_mask_integral(MaskSourceRandomLocation(m).data)
    zarr_integ = zarr.array(integ, chunks=(16, 16, 16))
    stock = _run_masked_locations(RandomLocation(min_masked=0.5, mask=m), m)
    zarred = _run_masked_locations(
        RandomLocation(min_masked=0.5, mask=m, mask_integral=zarr_integ), m
    )
    assert stock == zarred


def test_precomputed_mask_integral_zarr_store(tmp_path):
    # a zarr store on disk (opened read-only, chunks read lazily) -> identical
    zarr = pytest.importorskip("zarr")
    m = ArrayKey("MASK")
    integ = compute_mask_integral(MaskSourceRandomLocation(m).data)
    store = str(tmp_path / "mask_integral.zarr")
    zarr.save_array(store, integ)

    rl = RandomLocation(min_masked=0.5, mask=m, mask_integral=store)
    src = MaskSourceRandomLocation(m)
    with build(src + rl):
        np.testing.assert_array_equal(np.asarray(rl.mask_integral[:]), integ)

    stock = _run_masked_locations(RandomLocation(min_masked=0.5, mask=m), m)
    zarred = _run_masked_locations(
        RandomLocation(min_masked=0.5, mask=m, mask_integral=store), m
    )
    assert stock == zarred
