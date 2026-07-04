//! Experimental host-memory hooks for PRD 010 linear-memory snapshot research.
//!
//! This module is intentionally Unix-only and feature-gated behind `embedded` or
//! `preinit` because it implements Wasmtime's unsafe `MemoryCreator` surface. It
//! is not used by the normal shared-engine path.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use wasmtime::{LinearMemory, MemoryCreator, MemoryType};

/// Aggregate counters for host-managed Wasm linear memories.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostMemoryStats {
    /// Number of linear-memory allocations created.
    pub allocations: u64,
    /// Number of linear-memory allocations dropped.
    pub deallocations: u64,
    /// Currently mapped bytes, including guard regions.
    pub mapped_bytes: usize,
    /// Peak mapped bytes, including guard regions.
    pub peak_mapped_bytes: usize,
    /// Currently accessible bytes committed with read/write permissions.
    pub accessible_bytes: usize,
    /// Peak accessible bytes.
    pub peak_accessible_bytes: usize,
    /// Number of successful grow operations.
    pub grow_count: u64,
}

/// Metadata for one live host-managed linear memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostMemoryRegion {
    /// Stable region id allocated by [`HostMemoryTracker`].
    pub id: u64,
    /// Raw base address for diagnostics. Do not dereference outside quiescent
    /// snapshot code.
    pub base_addr: usize,
    /// Current Wasm byte size.
    pub byte_size: usize,
    /// Current byte capacity before relocation would be required.
    pub byte_capacity: usize,
    /// Currently page-accessible byte count.
    pub accessible_bytes: usize,
    /// Mapped bytes including guard region.
    pub mapped_bytes: usize,
    /// Guard bytes after the capacity region.
    pub guard_bytes: usize,
}

/// A copied quiescent snapshot of one host-managed linear memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostMemoryRegionSnapshot {
    /// Stable region id.
    pub id: u64,
    /// Capacity at capture time.
    pub byte_capacity: usize,
    /// Copied bytes for the current Wasm byte size.
    pub bytes: Vec<u8>,
}

/// Summary of a quiescent restore into live host-managed memories.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostMemoryRestoreStats {
    /// Number of regions restored.
    pub restored_regions: usize,
    /// Total bytes copied back into live memories.
    pub restored_bytes: usize,
}

#[derive(Debug)]
struct RegionRecord {
    base_addr: usize,
    byte_size: usize,
    byte_capacity: usize,
    accessible_bytes: usize,
    mapped_bytes: usize,
    guard_bytes: usize,
}

/// Tracks live linear memories allocated through [`TrackingMemoryCreator`].
#[derive(Debug, Default)]
pub struct HostMemoryTracker {
    next_id: AtomicU64,
    allocations: AtomicU64,
    deallocations: AtomicU64,
    mapped_bytes: AtomicUsize,
    peak_mapped_bytes: AtomicUsize,
    accessible_bytes: AtomicUsize,
    peak_accessible_bytes: AtomicUsize,
    grow_count: AtomicU64,
    regions: Mutex<HashMap<u64, Arc<Mutex<RegionRecord>>>>,
}

impl HostMemoryTracker {
    /// Return aggregate host-memory stats.
    #[must_use]
    pub fn stats(&self) -> HostMemoryStats {
        HostMemoryStats {
            allocations: self.allocations.load(Ordering::Relaxed),
            deallocations: self.deallocations.load(Ordering::Relaxed),
            mapped_bytes: self.mapped_bytes.load(Ordering::Relaxed),
            peak_mapped_bytes: self.peak_mapped_bytes.load(Ordering::Relaxed),
            accessible_bytes: self.accessible_bytes.load(Ordering::Relaxed),
            peak_accessible_bytes: self.peak_accessible_bytes.load(Ordering::Relaxed),
            grow_count: self.grow_count.load(Ordering::Relaxed),
        }
    }

    /// Return metadata for all currently live host-managed linear memories.
    #[must_use]
    pub fn live_regions(&self) -> Vec<HostMemoryRegion> {
        let regions = self
            .regions
            .lock()
            .expect("host memory region lock poisoned");

        regions
            .iter()
            .map(|(id, region)| {
                let region = region.lock().expect("host memory region lock poisoned");
                HostMemoryRegion {
                    id: *id,
                    base_addr: region.base_addr,
                    byte_size: region.byte_size,
                    byte_capacity: region.byte_capacity,
                    accessible_bytes: region.accessible_bytes,
                    mapped_bytes: region.mapped_bytes,
                    guard_bytes: region.guard_bytes,
                }
            })
            .collect()
    }

    /// Copy bytes from all currently live linear memories.
    ///
    /// The caller must only use this at a quiescent point where no Wasm frames
    /// are live and the store cannot re-enter guest code.
    #[allow(unsafe_code)]
    #[must_use]
    pub fn snapshot_live_regions(&self) -> Vec<HostMemoryRegionSnapshot> {
        let regions = self
            .regions
            .lock()
            .expect("host memory region lock poisoned");

        regions
            .iter()
            .map(|(id, region)| {
                let region = region.lock().expect("host memory region lock poisoned");
                let bytes = if region.byte_size == 0 {
                    Vec::new()
                } else {
                    // SAFETY: Regions are only removed while the LinearMemory is
                    // dropped, and this method holds the tracker lock while
                    // copying. The caller is responsible for quiescence so guest
                    // code cannot mutate or grow memory concurrently.
                    unsafe {
                        std::slice::from_raw_parts(region.base_addr as *const u8, region.byte_size)
                            .to_vec()
                    }
                };

                HostMemoryRegionSnapshot {
                    id: *id,
                    byte_capacity: region.byte_capacity,
                    bytes,
                }
            })
            .collect()
    }

    /// Copy snapshot bytes back into currently live linear memories.
    ///
    /// This is an in-place restore probe for PRD 010. The caller must only use
    /// this at a quiescent point with the same live instance whose memories were
    /// captured. It intentionally requires exact byte-size matches so the probe
    /// does not pretend to restore Wasmtime memory-size metadata, globals,
    /// tables, or host state.
    #[allow(unsafe_code)]
    pub fn restore_live_regions(
        &self,
        snapshots: &[HostMemoryRegionSnapshot],
    ) -> Result<HostMemoryRestoreStats, String> {
        let regions = self
            .regions
            .lock()
            .expect("host memory region lock poisoned");

        let mut restored = HostMemoryRestoreStats::default();
        for snapshot in snapshots {
            let region = regions
                .get(&snapshot.id)
                .ok_or_else(|| format!("linear-memory region {} is not live", snapshot.id))?;
            let region = region.lock().expect("host memory region lock poisoned");

            if snapshot.bytes.len() != region.byte_size {
                return Err(format!(
                    "linear-memory region {} size mismatch: snapshot={} live={}",
                    snapshot.id,
                    snapshot.bytes.len(),
                    region.byte_size
                ));
            }
            if snapshot.byte_capacity != region.byte_capacity {
                return Err(format!(
                    "linear-memory region {} capacity mismatch: snapshot={} live={}",
                    snapshot.id, snapshot.byte_capacity, region.byte_capacity
                ));
            }

            if !snapshot.bytes.is_empty() {
                // SAFETY: The region is live while held in the tracker map and
                // has the same byte_size as the snapshot. The caller guarantees
                // no guest code can run or mutate memory concurrently.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        snapshot.bytes.as_ptr(),
                        region.base_addr as *mut u8,
                        snapshot.bytes.len(),
                    );
                }
            }
            restored.restored_regions += 1;
            restored.restored_bytes += snapshot.bytes.len();
        }

        Ok(restored)
    }

    /// Negative metadata-only grow probe for fresh-session restore.
    ///
    /// This method is kept only to fail loudly for the path that PRD 010 proved
    /// unsafe: changing tracker metadata without updating Wasmtime's
    /// `VMMemoryDefinition.current_length`. Use [`restore_live_regions`] after
    /// creating a fresh session with [`TrackingMemoryCreator::new_with_initial_size_floor`].
    #[allow(unsafe_code)]
    pub fn restore_live_regions_growing(
        &self,
        snapshots: &[HostMemoryRegionSnapshot],
    ) -> Result<HostMemoryRestoreStats, String> {
        let regions = self
            .regions
            .lock()
            .expect("host memory region lock poisoned");

        let mut restored = HostMemoryRestoreStats::default();
        for snapshot in snapshots {
            let region = regions
                .get(&snapshot.id)
                .ok_or_else(|| format!("linear-memory region {} is not live", snapshot.id))?;
            let region = region.lock().expect("host memory region lock poisoned");

            if snapshot.byte_capacity != region.byte_capacity {
                return Err(format!(
                    "linear-memory region {} capacity mismatch: snapshot={} live={}",
                    snapshot.id, snapshot.byte_capacity, region.byte_capacity
                ));
            }

            let snapshot_size = snapshot.bytes.len();
            if snapshot_size > region.byte_capacity {
                return Err(format!(
                    "linear-memory region {} snapshot size {} exceeds live capacity {}",
                    snapshot.id, snapshot_size, region.byte_capacity
                ));
            }
            if snapshot_size < region.byte_size {
                return Err(format!(
                    "linear-memory region {} cannot shrink live memory: snapshot={} live={}",
                    snapshot.id, snapshot_size, region.byte_size
                ));
            }
            if snapshot_size > region.byte_size {
                return Err(
                    "metadata-only live-region grow is unsupported: it changes tracker state \
                     without updating Wasmtime VM memory length; create the restore target with \
                     TrackingMemoryCreator::new_with_initial_size_floor instead"
                        .to_string(),
                );
            }

            if !snapshot.bytes.is_empty() {
                // SAFETY: The region is live while held in the tracker map. It
                // has been grown to exactly the snapshot byte size above. The
                // caller guarantees no guest code can run or mutate memory
                // concurrently.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        snapshot.bytes.as_ptr(),
                        region.base_addr as *mut u8,
                        snapshot.bytes.len(),
                    );
                }
            }
            restored.restored_regions += 1;
            restored.restored_bytes += snapshot.bytes.len();
        }

        Ok(restored)
    }

    fn record_allocation(
        &self,
        base_addr: usize,
        byte_size: usize,
        byte_capacity: usize,
        accessible_bytes: usize,
        mapped_bytes: usize,
        guard_bytes: usize,
    ) -> (u64, Arc<Mutex<RegionRecord>>) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        self.allocations.fetch_add(1, Ordering::Relaxed);
        add_with_peak(&self.mapped_bytes, &self.peak_mapped_bytes, mapped_bytes);
        add_with_peak(
            &self.accessible_bytes,
            &self.peak_accessible_bytes,
            accessible_bytes,
        );

        let region = Arc::new(Mutex::new(RegionRecord {
            base_addr,
            byte_size,
            byte_capacity,
            accessible_bytes,
            mapped_bytes,
            guard_bytes,
        }));

        self.regions
            .lock()
            .expect("host memory region lock poisoned")
            .insert(id, Arc::clone(&region));

        (id, region)
    }

    fn record_growth_locked(
        &self,
        region: &mut RegionRecord,
        byte_size: usize,
        accessible_bytes: usize,
    ) {
        if accessible_bytes > region.accessible_bytes {
            add_with_peak(
                &self.accessible_bytes,
                &self.peak_accessible_bytes,
                accessible_bytes - region.accessible_bytes,
            );
        }
        region.byte_size = byte_size;
        region.accessible_bytes = accessible_bytes;
    }

    #[allow(unsafe_code)]
    fn grow_region_record(&self, region: &mut RegionRecord, new_size: usize) -> Result<(), String> {
        if new_size > region.byte_capacity {
            return Err(format!(
                "linear memory grow to {new_size} exceeds reserved capacity {}",
                region.byte_capacity
            ));
        }

        let page_size = page_size()?;
        let new_accessible = round_up(new_size, page_size)?;
        if new_accessible > region.accessible_bytes {
            // SAFETY: new_accessible is bounded by byte_capacity, and the range
            // starts at the mmap base owned by this live linear memory.
            let rc = unsafe {
                libc::mprotect(
                    region.base_addr as *mut libc::c_void,
                    new_accessible,
                    libc::PROT_READ | libc::PROT_WRITE,
                )
            };
            if rc != 0 {
                return Err(format!(
                    "mprotect failed: {}",
                    std::io::Error::last_os_error()
                ));
            }
        }

        self.record_growth_locked(region, new_size, new_accessible);
        self.grow_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn record_deallocation(&self, id: u64) {
        let Some(region) = self
            .regions
            .lock()
            .expect("host memory region lock poisoned")
            .remove(&id)
        else {
            return;
        };

        self.deallocations.fetch_add(1, Ordering::Relaxed);
        let region = region.lock().expect("host memory region lock poisoned");
        self.mapped_bytes
            .fetch_sub(region.mapped_bytes, Ordering::Relaxed);
        self.accessible_bytes
            .fetch_sub(region.accessible_bytes, Ordering::Relaxed);
    }
}

/// A Wasmtime `MemoryCreator` that allocates Unix mmap-backed linear memories
/// and records live-region metadata for quiescent snapshot experiments.
#[derive(Debug, Clone)]
pub struct TrackingMemoryCreator {
    tracker: Arc<HostMemoryTracker>,
    initial_size_floor: Option<usize>,
}

impl TrackingMemoryCreator {
    /// Create a new tracking memory creator and tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tracker: Arc::new(HostMemoryTracker::default()),
            initial_size_floor: None,
        }
    }

    /// Create a new tracking memory creator whose fresh allocations start at
    /// least `initial_size_floor` bytes.
    ///
    /// This is a PRD 010 restore spike hook. It lets the host instantiate a
    /// fresh session with a linear-memory shape compatible with a captured
    /// snapshot, without running guest code merely to grow memory.
    #[must_use]
    pub fn new_with_initial_size_floor(initial_size_floor: usize) -> Self {
        Self {
            tracker: Arc::new(HostMemoryTracker::default()),
            initial_size_floor: Some(initial_size_floor),
        }
    }

    /// Return the shared tracker.
    #[must_use]
    pub fn tracker(&self) -> Arc<HostMemoryTracker> {
        Arc::clone(&self.tracker)
    }
}

impl Default for TrackingMemoryCreator {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(unsafe_code)]
unsafe impl MemoryCreator for TrackingMemoryCreator {
    fn new_memory(
        &self,
        _ty: MemoryType,
        minimum: usize,
        maximum: Option<usize>,
        reserved_size_in_bytes: Option<usize>,
        guard_size_in_bytes: usize,
    ) -> Result<Box<dyn LinearMemory>, String> {
        let memory = TrackedLinearMemory::new(
            Arc::clone(&self.tracker),
            minimum,
            self.initial_size_floor,
            maximum,
            reserved_size_in_bytes,
            guard_size_in_bytes,
        )?;
        Ok(Box::new(memory))
    }
}

#[derive(Debug)]
struct TrackedLinearMemory {
    tracker: Arc<HostMemoryTracker>,
    id: u64,
    state: Arc<Mutex<RegionRecord>>,
    base: NonNull<u8>,
    mapped_bytes: usize,
}

#[allow(unsafe_code)]
unsafe impl Send for TrackedLinearMemory {}

#[allow(unsafe_code)]
unsafe impl Sync for TrackedLinearMemory {}

impl TrackedLinearMemory {
    #[allow(unsafe_code)]
    fn new(
        tracker: Arc<HostMemoryTracker>,
        minimum: usize,
        initial_size_floor: Option<usize>,
        maximum: Option<usize>,
        reserved_size_in_bytes: Option<usize>,
        guard_size_in_bytes: usize,
    ) -> Result<Self, String> {
        let page_size = page_size()?;
        let initial_size = initial_size_floor.unwrap_or(minimum).max(minimum);
        if let Some(maximum) = maximum
            && initial_size > maximum
        {
            return Err(format!(
                "linear memory initial size {initial_size} exceeds maximum {maximum}"
            ));
        }

        let mut byte_capacity = reserved_size_in_bytes
            .or(maximum)
            .unwrap_or(minimum)
            .max(initial_size);
        if let Some(maximum) = maximum {
            byte_capacity = byte_capacity.min(maximum).max(initial_size);
        }
        byte_capacity = round_up(byte_capacity, page_size)?;

        let guard_bytes = round_up(guard_size_in_bytes, page_size)?;
        let mapped_bytes = byte_capacity
            .checked_add(guard_bytes)
            .and_then(|bytes| {
                if bytes == 0 {
                    Some(page_size)
                } else {
                    Some(bytes)
                }
            })
            .ok_or_else(|| "linear memory mapping size overflow".to_string())?;

        // SAFETY: mmap is called with an anonymous private mapping. The returned
        // address is checked for failure and stored for munmap in Drop.
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped_bytes,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(format!(
                "mmap failed for {mapped_bytes} bytes: {}",
                std::io::Error::last_os_error()
            ));
        }
        let base =
            NonNull::new(raw.cast::<u8>()).ok_or_else(|| "mmap returned null".to_string())?;

        let accessible_bytes = round_up(initial_size, page_size)?;
        if accessible_bytes > 0 {
            // SAFETY: `base` refers to an mmap of at least `mapped_bytes`, and
            // accessible_bytes is bounded by byte_capacity.
            let rc = unsafe {
                libc::mprotect(
                    base.as_ptr().cast(),
                    accessible_bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                )
            };
            if rc != 0 {
                let err = std::io::Error::last_os_error();
                // SAFETY: base/mapped_bytes came from a successful mmap above.
                unsafe {
                    libc::munmap(base.as_ptr().cast(), mapped_bytes);
                }
                return Err(format!("mprotect failed: {err}"));
            }
        }

        let (id, state) = tracker.record_allocation(
            base.as_ptr() as usize,
            initial_size,
            byte_capacity,
            accessible_bytes,
            mapped_bytes,
            guard_bytes,
        );

        Ok(Self {
            tracker,
            id,
            state,
            base,
            mapped_bytes,
        })
    }
}

impl Drop for TrackedLinearMemory {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        self.tracker.record_deallocation(self.id);
        // SAFETY: base/mapped_bytes are owned by this LinearMemory and were
        // created by mmap in `new`.
        unsafe {
            libc::munmap(self.base.as_ptr().cast(), self.mapped_bytes);
        }
    }
}

#[allow(unsafe_code)]
unsafe impl LinearMemory for TrackedLinearMemory {
    fn byte_size(&self) -> usize {
        self.state
            .lock()
            .expect("host memory region lock poisoned")
            .byte_size
    }

    fn byte_capacity(&self) -> usize {
        self.state
            .lock()
            .expect("host memory region lock poisoned")
            .byte_capacity
    }

    fn grow_to(&mut self, new_size: usize) -> wasmtime::Result<()> {
        let mut state = self.state.lock().expect("host memory region lock poisoned");
        self.tracker
            .grow_region_record(&mut state, new_size)
            .map_err(wasmtime::Error::msg)?;
        Ok(())
    }

    fn as_ptr(&self) -> *mut u8 {
        self.base.as_ptr()
    }
}

fn add_with_peak(current: &AtomicUsize, peak: &AtomicUsize, delta: usize) {
    let new_current = current.fetch_add(delta, Ordering::Relaxed) + delta;
    let mut observed_peak = peak.load(Ordering::Relaxed);
    while new_current > observed_peak {
        match peak.compare_exchange_weak(
            observed_peak,
            new_current,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(next) => observed_peak = next,
        }
    }
}

#[allow(unsafe_code)]
fn page_size() -> Result<usize, String> {
    // SAFETY: sysconf is thread-safe and has no memory safety preconditions.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(format!(
            "sysconf(_SC_PAGESIZE) failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    usize::try_from(page_size).map_err(|_| "page size does not fit usize".to_string())
}

fn round_up(value: usize, align: usize) -> Result<usize, String> {
    debug_assert!(align.is_power_of_two());
    value
        .checked_add(align - 1)
        .map(|value| value & !(align - 1))
        .ok_or_else(|| "page alignment overflow".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(unsafe_code)]
    fn tracking_memory_creator_copies_live_region() {
        let creator = TrackingMemoryCreator::new();
        let tracker = creator.tracker();

        let mut memory = creator
            .new_memory(MemoryType::new(1, None), 4096, Some(8192), Some(8192), 4096)
            .expect("memory allocation should succeed");

        // SAFETY: This is a direct unit test of the host-memory implementation.
        // The memory is not attached to a running Wasm store and cannot be
        // concurrently accessed.
        unsafe {
            std::ptr::copy_nonoverlapping(b"rlm".as_ptr(), memory.as_ptr(), 3);
        }

        let snapshots = tracker.snapshot_live_regions();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(&snapshots[0].bytes[..3], b"rlm");
        assert_eq!(tracker.stats().allocations, 1);
        assert_eq!(tracker.stats().deallocations, 0);

        memory
            .grow_to(8192)
            .expect("grow should fit reserved capacity");
        assert_eq!(tracker.stats().grow_count, 1);

        drop(memory);
        assert_eq!(tracker.stats().deallocations, 1);
        assert!(tracker.live_regions().is_empty());
    }

    #[test]
    fn tracking_memory_creator_can_start_at_initial_size_floor() {
        let creator = TrackingMemoryCreator::new_with_initial_size_floor(8192);
        let tracker = creator.tracker();

        let memory = creator
            .new_memory(
                MemoryType::new(1, None),
                4096,
                Some(16384),
                Some(16384),
                4096,
            )
            .expect("memory allocation should succeed");

        assert_eq!(memory.byte_size(), 8192);
        assert_eq!(tracker.live_regions()[0].byte_size, 8192);
        assert!(tracker.live_regions()[0].accessible_bytes >= 8192);
    }
}
