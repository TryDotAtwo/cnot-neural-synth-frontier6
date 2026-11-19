use crate::device_state::DeviceStateBatch;
use crate::model::{DeviceFeatureEntry, DeviceFeatureStatus};
use crate::FeatureSpec;
use tch::{Device, Kind, Tensor};

fn empty(shape: &[i64], kind: Kind, device: Device) -> Result<Tensor, String> {
    let size = shape
        .iter()
        .try_fold(1i64, |size, &next| size.checked_mul(next))
        .ok_or("basis-gradient tensor extent overflow")?;
    size.checked_mul(if kind == Kind::Uint8 { 1 } else { 8 })
        .ok_or("basis-gradient tensor byte extent overflow")?;
    Tensor::f_empty(shape, (kind, device)).map_err(|e| e.to_string())
}

pub(super) fn append_entries(
    states: &DeviceStateBatch,
    specs: &[FeatureSpec],
    computed: &Tensor,
    entries: &mut Vec<DeviceFeatureEntry>,
) -> Result<(), String> {
    let b = states.batch_size();
    let n = i64::from(states.dimension());
    let w = n / 2;
    let t = n - w;
    let words = i64::from(states.dimension().div_ceil(64));
    let coefficient_words = (w + 63) / 64;
    let needs_pairs = specs
        .iter()
        .any(|s| s.id.family() == 'E' && matches!(s.id.ordinal(), 4 | 5));
    let needs_stats = specs
        .iter()
        .any(|s| s.id.family() == 'E' && s.id.ordinal() == 5);
    let count = w
        .checked_mul(w - 1)
        .ok_or("basis-gradient pair count overflow")?;
    if needs_stats {
        // This bound also prevents any partial sum from overflowing int64.
        count
            .checked_mul(t + 2)
            .ok_or("basis-gradient exact statistics exceed int64 capacity")?;
    }
    let device = states.tensor().device();
    let Device::Cuda(device_index) = device else {
        return Err("basis-gradient requires CUDA".into());
    };
    let scratch = empty(&[b, w, words + coefficient_words + 1], Kind::Int64, device)?;
    let coefficients = empty(&[b, t, w], Kind::Uint8, device)?;
    let raw_status = empty(&[b], Kind::Uint8, device)?;
    let gradients = empty(
        &[b, if needs_pairs { w } else { 0 }, w],
        Kind::Int64,
        device,
    )?;
    // Storage belongs to the current Torch stream; keep all scratch and outputs
    // alive until kernels and their tensor consumers have been queued.
    unsafe {
        crate::cuda::launch_basis_gradient(
            states.tensor().data_ptr().cast(),
            scratch.data_ptr().cast(),
            coefficients.data_ptr().cast(),
            raw_status.data_ptr().cast(),
            gradients.data_ptr().cast(),
            u32::try_from(b).map_err(|_| "basis-gradient batch exceeds CUDA range")?,
            states.dimension(),
            states.dimension().div_ceil(64),
            scratch.numel() as u64,
            coefficients.numel() as u64,
            gradients.numel() as u64,
            needs_pairs,
            crate::cuda::current_stream(device_index).map_err(str::to_owned)?,
        )?;
    }
    let available = if w < 2 {
        Tensor::zeros([b], (Kind::Bool, device))
    } else {
        raw_status.eq(1)
    };
    let inapplicable = computed.full_like(DeviceFeatureStatus::Inapplicable as i64);
    let unsupported = computed.full_like(DeviceFeatureStatus::Unsupported as i64);
    let status = computed.where_self(
        &available,
        &inapplicable.where_self(&raw_status.eq(0), &unsupported),
    );
    let off_diagonal = if needs_pairs {
        Some(
            Tensor::eye(w, (Kind::Int64, device))
                .eq(0)
                .unsqueeze(0)
                .expand([b, w, w], true),
        )
    } else {
        None
    };
    for spec in specs.iter().filter(|s| s.id.family() == 'E') {
        let values = match spec.id.ordinal() {
            1 => coefficients
                .sum_dim_intlist([1i64, 2].as_slice(), false, Kind::Int64)
                .to_kind(Kind::Float)
                .view([b, 1]),
            4 => Tensor::stack(
                &[
                    gradients.to_kind(Kind::Float),
                    off_diagonal
                        .as_ref()
                        .expect("E04 pairs")
                        .to_kind(Kind::Float),
                ],
                3,
            )
            .constant_pad_nd([0, 0, 0, t, 0, t]),
            5 if w < 2 => Tensor::zeros([b, 4], (Kind::Float, device)),
            5 => {
                let mask = off_diagonal.as_ref().expect("E05 pairs");
                let minimum = gradients
                    .where_self(mask, &gradients.full_like(i64::MAX))
                    .view([b, w * w])
                    .min_dim(1, false)
                    .0;
                let maximum = gradients
                    .where_self(mask, &gradients.full_like(i64::MIN))
                    .view([b, w * w])
                    .max_dim(1, false)
                    .0;
                let sum = gradients.sum_dim_intlist([1i64, 2].as_slice(), false, Kind::Int64);
                let negative =
                    gradients
                        .lt(0)
                        .sum_dim_intlist([1i64, 2].as_slice(), false, Kind::Int64);
                Tensor::stack(
                    &[
                        minimum.to_kind(Kind::Float),
                        (sum.to_kind(Kind::Double) / (count as f64)).to_kind(Kind::Float),
                        maximum.to_kind(Kind::Float),
                        negative.to_kind(Kind::Float) / (count as f64),
                    ],
                    1,
                )
            }
            _ => continue,
        };
        let mask = if spec.id.ordinal() == 4 {
            available.view([b, 1, 1, 1])
        } else {
            available.view([b, 1])
        };
        entries.push(DeviceFeatureEntry::new(
            spec.clone(),
            values.where_self(&mask, &Tensor::zeros_like(&values)),
            status.shallow_clone(),
            None,
            None,
        ));
    }
    Ok(())
}
