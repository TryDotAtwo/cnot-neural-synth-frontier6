use tch::{Kind, Tensor};

/// Exact equality of every packed word. Stable sorts keep the closed set first
/// and preserve generation order among equal children, including across parents.
pub(super) fn first_unseen(closed: &Tensor, children: &Tensor) -> Tensor {
    let previous = closed.size()[0];
    let count = children.size()[0];
    let keys = Tensor::cat(&[closed, children], 0).flatten(1, -1);
    let total = previous + count;
    let device = keys.device();
    let mut order = Tensor::arange(total, (Kind::Int64, device));
    for col in (0..keys.size()[1]).rev() {
        let column = keys.select(1, col).index_select(0, &order);
        order = order.index_select(0, &column.argsort_stable(true, 0, false));
    }
    let sorted = keys.index_select(0, &order);
    let different = sorted
        .narrow(0, 1, total - 1)
        .ne_tensor(&sorted.narrow(0, 0, total - 1))
        .any_dim(1, false);
    let first = Tensor::cat(&[Tensor::ones([1], (Kind::Bool, device)), different], 0);
    first
        .zeros_like()
        .scatter(0, &order, &first)
        .narrow(0, previous, count)
}

pub(super) struct Frontier {
    pub(super) rows: Tensor,
    pub(super) words: Tensor,
    pub(super) lex_rank: Tensor,
    pub(super) live: Tensor,
}

impl Frontier {
    /// A full-alphabet one-step goal test precedes every approximate shortlist.
    /// Among hits of equal length use the lexicographically first complete word.
    pub(super) fn nearest(&self, identity: &Tensor) -> (Tensor, Tensor) {
        let shape = self.rows.size();
        let (p, n, w) = (shape[0], shape[1], shape[2]);
        let diff = self.rows.bitwise_xor_tensor(identity);
        let changed = diff.ne(0).any_dim(2, false);
        let target = changed.to_kind(Kind::Int64).argmax(1, false);
        let needed = diff.gather(1, &target.view([p, 1, 1]).expand([p, 1, w], true), false);
        let physical = Tensor::arange(n, (Kind::Int64, self.rows.device()));
        let matches = self
            .rows
            .eq_tensor(&needed)
            .all_dim(2, false)
            .logical_and(&physical.unsqueeze(0).ne_tensor(&target.unsqueeze(1)));
        let goal = self
            .live
            .logical_and(
                &changed
                    .sum_dim_intlist([1i64].as_slice(), false, Kind::Int64)
                    .eq(1),
            )
            .logical_and(&matches.any_dim(1, false));
        let control = matches.to_kind(Kind::Int64).argmax(1, false);
        let action = &control * (n - 1) + &target - target.gt_tensor(&control).to_kind(Kind::Int64);
        let parent = self
            .lex_rank
            .masked_fill(&goal.logical_not(), i64::MAX)
            .argmin(0, false)
            .view([1]);
        let prefix = self.words.index_select(0, &parent).view([-1]);
        (
            goal.any(),
            Tensor::cat(&[prefix, action.index_select(0, &parent)], 0),
        )
    }

    pub(super) fn expand(&self, ranked: &Tensor, shortlist: i64) -> (Tensor, Tensor, Tensor) {
        let shape = self.rows.size();
        let (p, n, w) = (shape[0], shape[1], shape[2]);
        let action = ranked.reshape([-1]);
        let parent = Tensor::arange(p, (Kind::Int64, self.rows.device()))
            .unsqueeze(1)
            .expand([p, shortlist], true)
            .reshape([-1]);
        let control = action.floor_divide_scalar(n - 1);
        let slot = action.remainder(n - 1);
        let target = &slot + slot.ge_tensor(&control).to_kind(Kind::Int64);
        let base = self.rows.index_select(0, &parent);
        let gather = |v: &Tensor| v.view([-1, 1, 1]).expand([-1, 1, w], true);
        let target = gather(&target);
        let changed = base
            .gather(1, &target, false)
            .bitwise_xor_tensor(&base.gather(1, &gather(&control), false));
        (base.scatter(1, &target, &changed), parent, action)
    }

    /// Score ties are ordered by the full physical word. At a fixed depth the
    /// pair (parent's lexicographic rank, physical action) orders those words.
    pub(super) fn retain(
        &self,
        children: &Tensor,
        parent: &Tensor,
        action: &Tensor,
        scores: &Tensor,
        live: &Tensor,
        width: i64,
        actions: i64,
        root: &Tensor,
    ) -> Self {
        let lex = self.lex_rank.index_select(0, parent) * actions + action;
        let by_word = lex.argsort_stable(true, 0, false);
        let scores = scores
            .masked_fill(&live.logical_not(), f64::INFINITY)
            .index_select(0, &by_word);
        let selected = by_word.index_select(
            0,
            &scores.argsort_stable(true, 0, false).narrow(0, 0, width),
        );
        let kept = live.index_select(0, &selected);
        let parent = parent.index_select(0, &selected);
        let words = Tensor::cat(
            &[
                self.words.index_select(0, &parent),
                action.index_select(0, &selected).unsqueeze(1),
            ],
            1,
        );
        let lex_rank = lex
            .index_select(0, &selected)
            .argsort_stable(true, 0, false)
            .argsort_stable(true, 0, false);
        let rows = children
            .index_select(0, &selected)
            .where_self(&kept.view([-1, 1, 1]), root);
        Self {
            rows,
            words,
            lex_rank,
            live: kept,
        }
    }
}
