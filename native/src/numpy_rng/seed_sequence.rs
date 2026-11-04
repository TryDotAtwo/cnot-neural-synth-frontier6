const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;

pub(super) fn state_and_increment(seed: u64) -> (u128, u128) {
    let mut entropy = vec![seed as u32];
    let high = (seed >> 32) as u32;
    if high != 0 {
        entropy.push(high);
    }
    let pool = seed_sequence_pool(&entropy);
    let mut hash_const = INIT_B;
    let mut words = [0u32; 8];
    for (index, word) in words.iter_mut().enumerate() {
        let value = pool[index % pool.len()] ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        let mixed = value.wrapping_mul(hash_const);
        *word = mixed ^ (mixed >> 16);
    }

    let initstate_high = u64::from(words[0]) | (u64::from(words[1]) << 32);
    let initstate_low = u64::from(words[2]) | (u64::from(words[3]) << 32);
    let initseq_high = u64::from(words[4]) | (u64::from(words[5]) << 32);
    let initseq_low = u64::from(words[6]) | (u64::from(words[7]) << 32);
    let state = (u128::from(initstate_high) << 64) | u128::from(initstate_low);
    let sequence = (u128::from(initseq_high) << 64) | u128::from(initseq_low);
    let increment = sequence.wrapping_shl(1) | 1;
    (state, increment)
}

fn seed_sequence_pool(entropy: &[u32]) -> [u32; 4] {
    let mut pool = [0u32; 4];
    let mut hash_const = INIT_A;
    for (index, value) in pool.iter_mut().enumerate() {
        *value = hash_mix(entropy.get(index).copied().unwrap_or(0), &mut hash_const);
    }
    for source in 0..pool.len() {
        let source_value = pool[source];
        for (destination, value) in pool.iter_mut().enumerate() {
            if source != destination {
                let mixed = hash_mix(source_value, &mut hash_const);
                *value = mix(*value, mixed);
            }
        }
    }
    for &extra in entropy.iter().skip(pool.len()) {
        for value in &mut pool {
            let mixed = hash_mix(extra, &mut hash_const);
            *value = mix(*value, mixed);
        }
    }
    pool
}

fn hash_mix(mut value: u32, hash_const: &mut u32) -> u32 {
    value ^= *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> 16)
}

fn mix(left: u32, right: u32) -> u32 {
    let value = MIX_MULT_L
        .wrapping_mul(left)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(right));
    value ^ (value >> 16)
}
