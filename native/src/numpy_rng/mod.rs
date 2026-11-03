//! NumPy default_rng(u64) compatibility for source-defined Monte Carlo inputs.

mod pcg64;
mod poisson;
mod seed_sequence;

#[cfg(test)]
mod tests;

pub(crate) use pcg64::NumpyRng;
