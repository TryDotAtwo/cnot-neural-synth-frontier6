use super::{FAMILY_COUNTS, FEATURE_COUNT};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FeatureId {
    pub(super) family: u8,
    pub(super) ordinal: u8,
}

impl FeatureId {
    pub fn new(family: u8, ordinal: u8) -> Option<Self> {
        FAMILY_COUNTS
            .iter()
            .any(|(known, count)| *known == family && (1..=*count).contains(&ordinal))
            .then_some(Self { family, ordinal })
    }

    pub fn from_index(index: usize) -> Option<Self> {
        if index >= FEATURE_COUNT {
            return None;
        }
        let mut offset = 0usize;
        for (family, count) in FAMILY_COUNTS {
            let end = offset + usize::from(count);
            if index < end {
                return Some(Self {
                    family,
                    ordinal: (index - offset + 1) as u8,
                });
            }
            offset = end;
        }
        None
    }

    pub fn family(self) -> char {
        char::from(self.family)
    }

    pub fn ordinal(self) -> u8 {
        self.ordinal
    }

    pub fn index(self) -> usize {
        let mut offset = 0usize;
        for (family, count) in FAMILY_COUNTS {
            if family == self.family {
                return offset + usize::from(self.ordinal - 1);
            }
            offset += usize::from(count);
        }
        unreachable!("FeatureId is validated on construction")
    }
}

impl fmt::Display for FeatureId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:02}", char::from(self.family), self.ordinal)
    }
}
