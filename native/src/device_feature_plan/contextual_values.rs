use super::contexts::PreparedContexts;
use crate::device_state::DeviceStateBatch;
use crate::model::{DeviceFeatureEntry, DeviceFeatureStatus};
use crate::FeatureSpec;
use tch::{Kind, Tensor};

impl PreparedContexts {
    pub(super) fn append_entries(
        &self,
        states: &DeviceStateBatch,
        specs: &[FeatureSpec],
        computed: &Tensor,
        entries: &mut Vec<DeviceFeatureEntry>,
        cached_f08: Option<&DeviceFeatureEntry>,
    ) -> Result<(), String> {
        let active = |family, ordinals: &[u8]| super::selected(specs, family, ordinals);
        let cuts = self
            .cuts
            .as_ref()
            .filter(|_| active('C', &[2, 3, 4]))
            .map(|c| states.labeled_cut_features(c))
            .transpose()?;
        let shear = (self.partial_shear && active('C', &[5]))
            .then(|| states.partial_shear_features())
            .transpose()?;
        let intersections = self
            .intersections
            .as_ref()
            .filter(|_| active('C', &[6, 7]))
            .map(|c| states.restricted_intersection_features(c))
            .transpose()?;
        let increments = self
            .increments
            .as_ref()
            .filter(|_| active('C', &[9]))
            .map(|c| states.rank_increment_features(c))
            .transpose()?;
        let residual = self
            .residual
            .as_ref()
            .filter(|_| active('C', &[10]))
            .map(|c| states.selected_residual_features(c))
            .transpose()?;
        let nodes = if active('D', &[5]) {
            Some(states.block_parity_d05(self.blocks.as_ref().expect("bound D05 context"))?)
        } else {
            None
        };
        let groups = if active('D', &[8]) {
            Some(states.block_parity_d08(self.blocks.as_ref().expect("bound D08 context"))?)
        } else {
            None
        };
        let boundary = self
            .boundary
            .as_ref()
            .filter(|_| active('C', &[8]) || active('I', &[1, 2, 3, 4, 11]))
            .map(|c| {
                if self.boundary_limited {
                    Ok(Tensor::zeros(
                        [states.batch_size(), 1],
                        (Kind::Float, computed.device()),
                    ))
                } else {
                    states.matrix_boundary_c08(c)
                }
            })
            .transpose()?;
        let limited = computed.full_like(DeviceFeatureStatus::ResourceLimit as i64);
        let group_entry =
            |spec: &FeatureSpec, values: &Tensor, mask: &Tensor, incidence: &Tensor| {
                DeviceFeatureEntry::new(
                    spec.clone(),
                    values.shallow_clone(),
                    computed.shallow_clone(),
                    Some(mask.shallow_clone()),
                    Some(incidence.shallow_clone()),
                )
            };
        for spec in specs {
            let entry = match (spec.id.family(), spec.id.ordinal()) {
                ('C', 2..=4) => {
                    let value = cuts.as_ref().expect("bound cuts");
                    group_entry(
                        spec,
                        value.feature(spec.id).expect("C02-C04"),
                        value.token_valid(),
                        value.incidence(),
                    )
                }
                ('C', 5) => {
                    let value = shear.as_ref().expect("bound shear producer");
                    let available = value.status().eq(1);
                    let inapplicable = computed.full_like(DeviceFeatureStatus::Inapplicable as i64);
                    let unsupported = computed.full_like(DeviceFeatureStatus::Unsupported as i64);
                    let unavailable = inapplicable.where_self(&value.status().eq(0), &unsupported);
                    DeviceFeatureEntry::new(
                        spec.clone(),
                        value.feature(spec.id).expect("C05").shallow_clone(),
                        computed.where_self(&available, &unavailable),
                        None,
                        None,
                    )
                }
                ('C', 6..=7) => {
                    let value = intersections.as_ref().expect("bound intersections");
                    group_entry(
                        spec,
                        value.feature(spec.id).expect("C06-C07"),
                        value.token_valid(),
                        value.incidence(),
                    )
                }
                ('C', 8) => DeviceFeatureEntry::new(
                    spec.clone(),
                    boundary.as_ref().expect("bound boundary").shallow_clone(),
                    if self.boundary_limited {
                        limited.shallow_clone()
                    } else {
                        computed.shallow_clone()
                    },
                    None,
                    None,
                ),
                ('C', 9) => {
                    let value = increments.as_ref().expect("bound increments");
                    group_entry(
                        spec,
                        value.feature(spec.id).expect("C09"),
                        value.token_valid(),
                        value.incidence(),
                    )
                }
                ('C', 10) => {
                    let value = residual.as_ref().expect("bound residual actions");
                    group_entry(
                        spec,
                        value.feature(spec.id).expect("C10"),
                        value.token_valid(),
                        value.incidence(),
                    )
                }
                ('D', 5) => {
                    let value = nodes.as_ref().expect("bound D05");
                    DeviceFeatureEntry::new(
                        spec.clone(),
                        value.feature(spec.id).expect("D05").squeeze_dim(1),
                        computed.where_self(&value.status().squeeze_dim(1).eq(1), &limited),
                        None,
                        None,
                    )
                }
                ('D', 8) => {
                    let value = groups.as_ref().expect("bound D08");
                    // The reference producer is unavailable for the entire
                    // sample if any physical block exceeds its pattern budget.
                    let available = value.status().eq(1).all_dim(1, false);
                    let payload = value.feature(spec.id).expect("D08");
                    let mask = value
                        .token_valid()
                        .to_kind(Kind::Bool)
                        .logical_and(&available.unsqueeze(1));
                    DeviceFeatureEntry::new(
                        spec.clone(),
                        payload
                            .where_self(&available.view([-1, 1, 1]), &Tensor::zeros_like(payload)),
                        computed.where_self(&available, &limited),
                        Some(mask),
                        Some(
                            value.incidence() * available.to_kind(Kind::Float).view([-1, 1, 1, 1]),
                        ),
                    )
                }
                _ => continue,
            };
            entries.push(entry);
        }
        if let Some(histogram) = self
            .block_histogram
            .as_ref()
            .filter(|_| active('D', &[1, 2, 4, 6, 7]))
        {
            histogram.append_entries(
                states,
                self.blocks.as_ref().expect("bound histogram context"),
                specs,
                computed,
                entries,
            )?;
        }
        if self.basis_gradient && active('E', &[1, 4, 5]) {
            super::basis_gradient::append_entries(states, specs, computed, entries)?;
        }
        if let Some(plan) = self.guarded_pair.as_ref().filter(|_| active('F', &[2, 6])) {
            let output =
                crate::stage_profile::measure("feature_F02_F06", || plan.evaluate(states))?;
            for spec in specs
                .iter()
                .filter(|s| s.id.family() == 'F' && matches!(s.id.ordinal(), 2 | 6))
            {
                let group = spec.id.ordinal() == 6;
                entries.push(DeviceFeatureEntry::new(
                    spec.clone(),
                    if group { output.f06() } else { output.f02() }.shallow_clone(),
                    output.status().shallow_clone(),
                    group.then(|| output.token_valid().shallow_clone()),
                    group.then(|| output.incidence().shallow_clone()),
                ));
            }
        }
        if let Some(plan) = self.zero_excess.as_ref().filter(|_| active('F', &[7, 8])) {
            let output = crate::stage_profile::measure("feature_F07", || plan.evaluate(states))?;
            for spec in specs
                .iter()
                .filter(|s| s.id.family() == 'F' && s.id.ordinal() == 7)
            {
                entries.push(DeviceFeatureEntry::new(
                    spec.clone(),
                    output.feature().shallow_clone(),
                    output.status().shallow_clone(),
                    None,
                    None,
                ));
            }
            if let Some(one_excess) = self.one_excess.as_ref().filter(|_| active('F', &[8])) {
                // F08-only and F07+F08 selections each compute the prerequisite
                // once. Both tensors remain private through queued consumers.
                let spec = specs
                    .iter()
                    .find(|s| (s.id.family(), s.id.ordinal()) == ('F', 8))
                    .expect("selected F08");
                let (values, status) = if let Some(cached) = cached_f08 {
                    if cached.spec() != spec
                        || cached.values().size() != [states.batch_size(), spec.channels as i64]
                        || cached.status().size() != [states.batch_size()]
                        || cached.values().device() != states.tensor().device()
                        || cached.status().device() != states.tensor().device()
                    {
                        return Err("cached root F08 differs from the feature contract".into());
                    }
                    (cached.values().shallow_clone(), cached.status().shallow_clone())
                } else {
                    let result = crate::stage_profile::measure("feature_F08", || {
                        one_excess.evaluate_with_zero_excess(states, &output)
                    })?;
                    (result.feature().shallow_clone(), result.status().shallow_clone())
                };
                entries.push(DeviceFeatureEntry::new(spec.clone(), values, status, None, None));
            }
        }
        if let Some(plan) = self
            .template_response
            .as_ref()
            .filter(|_| active('E', &[2, 3, 6, 7, 8, 9]))
        {
            let result = plan.evaluate(states)?;
            for spec in specs
                .iter()
                .filter(|s| s.id.family() == 'E' && [2, 3, 6, 7, 8, 9].contains(&s.id.ordinal()))
            {
                let group = spec.id.ordinal() == 6;
                entries.push(DeviceFeatureEntry::new(
                    spec.clone(),
                    result
                        .feature(spec.id.ordinal())
                        .expect("selected template response"),
                    result.status().shallow_clone(),
                    group.then(|| result.token_valid().shallow_clone()),
                    group.then(|| result.incidence().shallow_clone()),
                ));
            }
        }
        if let Some(plan) = self
            .paid_basis
            .as_ref()
            .filter(|_| active('D', &[9, 10, 11, 12, 13, 14]))
        {
            let result = plan.evaluate(states)?;
            for spec in specs
                .iter()
                .filter(|s| s.id.family() == 'D' && (9..=14).contains(&s.id.ordinal()))
            {
                let group = spec.id.ordinal() != 14;
                entries.push(DeviceFeatureEntry::new(
                    spec.clone(),
                    result.feature(spec.id.ordinal()).expect("selected D09-D14"),
                    result.status().shallow_clone(),
                    group.then(|| result.token_valid().shallow_clone()),
                    group.then(|| result.incidence().shallow_clone()),
                ));
            }
        }
        if let Some(stage26) = self.stage26.as_ref().filter(|_| active('F', &[3, 4, 5])) {
            stage26.append_entries(states, specs, entries)?;
        }
        if let Some(frequency) = self.frequency.as_ref().filter(|_| active('I', &[7])) {
            frequency.append_entries(states, specs, entries)?;
        }
        if let Some(relation) = self.shared_relation.as_ref().filter(|_| active('I', &[6])) {
            relation.append_entries(states, specs, entries)?;
        }
        if let Some(bilinear) = self.bilinear.as_ref().filter(|_| active('I', &[5])) {
            bilinear.append_entries(states, specs, entries)?;
        }
        if let Some(packet) = self.positive_packet.as_ref().filter(|_| active('I', &[8])) {
            packet.append_entries(states, specs, entries)?;
        }
        if let Some(quadratic) = self.quadratic.as_ref().filter(|_| active('I', &[9])) {
            let response = quadratic.compile(states)?;
            let spec = specs
                .iter()
                .find(|s| (s.id.family(), s.id.ordinal()) == ('I', 9))
                .expect("selected I09");
            entries.push(DeviceFeatureEntry::new(
                spec.clone(),
                response.values().shallow_clone(),
                response.status().shallow_clone(),
                Some(response.token_valid().shallow_clone()),
                Some(response.incidence().shallow_clone()),
            ));
        }
        if let Some(response) = self
            .endpoint_response
            .as_ref()
            .filter(|_| active('I', &[10]))
        {
            let result = response.evaluate(states)?;
            let spec = specs
                .iter()
                .find(|s| (s.id.family(), s.id.ordinal()) == ('I', 10))
                .expect("selected I10");
            entries.push(DeviceFeatureEntry::new(
                spec.clone(),
                result.values().shallow_clone(),
                result.status().shallow_clone(),
                None,
                None,
            ));
        }
        if let Some(compiler) = self
            .general_compiler
            .as_ref()
            .filter(|_| active('F', &[9, 10]))
        {
            let words = compiler.compile(states)?;
            for spec in specs
                .iter()
                .filter(|s| s.id.family() == 'F' && matches!(s.id.ordinal(), 9 | 10))
            {
                let values = if spec.id.ordinal() == 9 {
                    words.f09()
                } else {
                    words.f10()
                };
                entries.push(DeviceFeatureEntry::new(
                    spec.clone(),
                    values.shallow_clone(),
                    words.status().shallow_clone(),
                    None,
                    None,
                ));
            }
        }
        if let Some(relation) = self
            .boundary_relation
            .as_ref()
            .filter(|_| active('I', &[1, 2, 3, 4, 11]))
        {
            relation.append_entries(
                states,
                specs,
                computed,
                boundary.as_ref().expect("bound boundary"),
                self.boundary_limited,
                entries,
            )?;
        }
        Ok(())
    }
}
