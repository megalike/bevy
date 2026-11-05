//! Types for controlling batching behavior during parallel processing.

use core::ops::Range;

/// Dictates how a parallel operation chunks up large quantities
/// during iteration.
///
/// A parallel query will chunk up large tables and archetypes into
/// chunks of at most a certain batch size. Similarly, a parallel event
/// reader will chunk up the remaining events.
///
/// By default, this batch size is automatically determined by dividing
/// the size of the largest matched archetype by the number
/// of threads (rounded up), but is never smaller than
/// [`BatchingStrategy::DEFAULT_MIN_BATCH_SIZE`]. This attempts to minimize the
/// overhead of scheduling tasks onto multiple threads, but assumes each entity
/// has roughly the same amount of work to be done, which may not hold true in
/// every workload. For a few items with a lot of work each, lower the minimum
/// with [`BatchingStrategy::min_batch_size`].
///
/// See [`Query::par_iter`], [`MessageReader::par_read`] for more information.
///
/// [`Query::par_iter`]: crate::system::Query::par_iter
/// [`MessageReader::par_read`]: crate::message::MessageReader::par_read
#[derive(Clone, Debug)]
pub struct BatchingStrategy {
    /// The upper and lower limits for a batch of items.
    ///
    /// Setting the bounds to the same value will result in a fixed
    /// batch size.
    ///
    /// Defaults to `[DEFAULT_MIN_BATCH_SIZE, usize::MAX]`.
    ///
    /// [`DEFAULT_MIN_BATCH_SIZE`]: Self::DEFAULT_MIN_BATCH_SIZE
    pub batch_size_limits: Range<usize>,
    /// The number of batches per thread in the [`ComputeTaskPool`].
    /// Increasing this value will decrease the batch size, which may
    /// increase the scheduling overhead for the iteration.
    ///
    /// Defaults to 1.
    ///
    /// [`ComputeTaskPool`]: bevy_tasks::ComputeTaskPool
    pub batches_per_thread: usize,
}

impl Default for BatchingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchingStrategy {
    /// The smallest batch the default strategy hands to a task.
    ///
    /// Spawning a task and waiting for it costs microseconds, more than the
    /// per-item work of most systems, so a small iteration runs as one task
    /// instead of one task per thread.
    // MPS: measured on a 24-thread compute pool (reviews/2026-11-05-task-pools.md): 64 recovers less of
    // the empty scene's overhead, 4096 loses the gain of large crowds.
    pub const DEFAULT_MIN_BATCH_SIZE: usize = 256;

    /// Creates a new default batching strategy, with batches of at least
    /// [`DEFAULT_MIN_BATCH_SIZE`](Self::DEFAULT_MIN_BATCH_SIZE) items.
    pub const fn new() -> Self {
        Self {
            batch_size_limits: Self::DEFAULT_MIN_BATCH_SIZE..usize::MAX,
            batches_per_thread: 1,
        }
    }

    /// Declares a batching strategy with a fixed batch size.
    pub const fn fixed(batch_size: usize) -> Self {
        Self {
            batch_size_limits: batch_size..batch_size,
            batches_per_thread: 1,
        }
    }

    /// Configures the minimum allowed batch size of this instance.
    pub const fn min_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size_limits.start = batch_size;
        self
    }

    /// Configures the maximum allowed batch size of this instance.
    ///
    /// Also lowers the minimum batch size to `batch_size` if it is larger.
    pub const fn max_batch_size(mut self, batch_size: usize) -> Self {
        if self.batch_size_limits.start > batch_size {
            self.batch_size_limits.start = batch_size;
        }
        self.batch_size_limits.end = batch_size;
        self
    }

    /// Configures the number of batches to assign to each thread for this instance.
    pub fn batches_per_thread(mut self, batches_per_thread: usize) -> Self {
        assert!(
            batches_per_thread > 0,
            "The number of batches per thread must be non-zero."
        );
        self.batches_per_thread = batches_per_thread;
        self
    }

    /// Calculate the batch size according to the given thread count and max item count.
    /// The count is provided as a closure so that it can be calculated only if needed.
    ///
    /// # Panics
    ///
    /// Panics if `thread_count` is 0.
    #[inline]
    pub fn calc_batch_size(&self, max_items: impl FnOnce() -> usize, thread_count: usize) -> usize {
        if self.batch_size_limits.is_empty() {
            return self.batch_size_limits.start;
        }
        assert!(
            thread_count > 0,
            "Attempted to run parallel iteration with an empty TaskPool"
        );
        let batches = thread_count * self.batches_per_thread;
        // Round up to the nearest batch size.
        let batch_size = max_items().div_ceil(batches);
        batch_size.clamp(self.batch_size_limits.start, self.batch_size_limits.end)
    }
}

#[cfg(test)]
mod tests {
    use super::BatchingStrategy;

    #[test]
    fn default_has_a_minimum_batch_size() {
        let strategy = BatchingStrategy::new();
        let min = BatchingStrategy::DEFAULT_MIN_BATCH_SIZE;
        assert_eq!(strategy.calc_batch_size(|| 100, 24), min);
        assert_eq!(strategy.calc_batch_size(|| 24 * min, 24), min);
        assert_eq!(strategy.calc_batch_size(|| 24 * min * 2, 24), min * 2);
    }

    #[test]
    fn min_batch_size_opts_out() {
        let strategy = BatchingStrategy::new().min_batch_size(1);
        assert_eq!(strategy.calc_batch_size(|| 100, 24), 5);
        assert_eq!(strategy.calc_batch_size(|| 10, 24), 1);
    }

    #[test]
    fn max_batch_size_below_the_minimum() {
        let strategy = BatchingStrategy::new().max_batch_size(64);
        assert_eq!(strategy.calc_batch_size(|| 100, 24), 64);
        assert_eq!(strategy.calc_batch_size(|| 100_000, 24), 64);
        let strategy = BatchingStrategy::new().min_batch_size(1).max_batch_size(64);
        assert_eq!(strategy.calc_batch_size(|| 100, 24), 5);
    }

    #[test]
    fn fixed_is_unchanged() {
        assert_eq!(
            BatchingStrategy::fixed(32).calc_batch_size(|| 100_000, 24),
            32
        );
    }
}
