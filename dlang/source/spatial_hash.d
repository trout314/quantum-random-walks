/**
 * spatial_hash.d — Spatial hash table for 3D site lookup.
 *
 * Maps 3D positions (discretized to integer keys) to site IDs.
 * The interface allows swapping implementations without changing
 * the lattice or walk code.
 */
module spatial_hash;

import geometry : Vec3;

/// Discretized 3D position key.
struct PosKey {
    long kx, ky, kz;
}

/// Discretize a Vec3 position to integer keys.
PosKey posKey(Vec3 p) {
    enum double tol = 1e-7;
    import std.math : round;
    return PosKey(
        cast(long) round(p.x / tol),
        cast(long) round(p.y / tol),
        cast(long) round(p.z / tol),
    );
}

/// Interface for spatial hash tables mapping positions to site IDs.
interface SpatialHash {
    /// Find the site ID at a position, or -1 if not present.
    int find(Vec3 pos);

    /// Insert a position with a given site ID. Returns true on success.
    bool insert(Vec3 pos, int id);

    /// Remove the entry at a position (tombstone). Returns true if found.
    bool remove(Vec3 pos);

    /// Number of entries currently stored.
    int count() const;
}

/// Open-addressing hash table with linear probing and tombstone support.
class OpenAddressingHash : SpatialHash {
    private {
        struct Entry {
            long kx, ky, kz;
            int id = EMPTY;
        }

        enum int EMPTY = -1;
        enum int TOMBSTONE = -2;

        Entry[] table;
        uint mask;
        int _count;
    }

    this(int bits) {
        assert(bits > 0 && bits <= 30);
        uint size = 1u << bits;
        mask = size - 1;
        table = new Entry[size];
        _count = 0;
    }

    private static uint hash(long kx, long ky, long kz) {
        return cast(uint)(
            (cast(ulong)(kx * 73856093L)
           ^ cast(ulong)(ky * 19349663L)
           ^ cast(ulong)(kz * 83492791L))
        );
    }

    int find(Vec3 pos) {
        auto k = posKey(pos);
        uint h = hash(k.kx, k.ky, k.kz);
        foreach (probe; 0 .. cast(int) table.length) {
            uint i = (h + probe) & mask;
            if (table[i].id == EMPTY) return -1;
            if (table[i].id == TOMBSTONE) continue;
            if (table[i].kx == k.kx && table[i].ky == k.ky && table[i].kz == k.kz)
                return table[i].id;
        }
        return -1;
    }

    bool insert(Vec3 pos, int id) {
        auto k = posKey(pos);
        uint h = hash(k.kx, k.ky, k.kz);
        int firstTomb = -1;
        foreach (probe; 0 .. cast(int) table.length) {
            uint i = (h + probe) & mask;
            if (table[i].id == TOMBSTONE) {
                if (firstTomb < 0) firstTomb = cast(int) i;
                continue;
            }
            if (table[i].id == EMPTY) {
                uint slot = (firstTomb >= 0) ? cast(uint) firstTomb : i;
                table[slot] = Entry(k.kx, k.ky, k.kz, id);
                _count++;
                return true;
            }
            if (table[i].kx == k.kx && table[i].ky == k.ky && table[i].kz == k.kz)
                return false;  // already exists
        }
        return false;  // table full
    }

    bool remove(Vec3 pos) {
        auto k = posKey(pos);
        uint h = hash(k.kx, k.ky, k.kz);
        foreach (probe; 0 .. cast(int) table.length) {
            uint i = (h + probe) & mask;
            if (table[i].id == EMPTY) return false;
            if (table[i].id == TOMBSTONE) continue;
            if (table[i].kx == k.kx && table[i].ky == k.ky && table[i].kz == k.kz) {
                table[i].id = TOMBSTONE;
                _count--;
                return true;
            }
        }
        return false;
    }

    int count() const { return _count; }
}

// ---- D unit tests ----

unittest {
    // Basic insert and find
    auto h = new OpenAddressingHash(10);
    assert(h.find(Vec3(1, 2, 3)) == -1);
    assert(h.insert(Vec3(1, 2, 3), 42));
    assert(h.find(Vec3(1, 2, 3)) == 42);
    assert(h.count == 1);
}

unittest {
    // Duplicate insert rejected
    auto h = new OpenAddressingHash(10);
    assert(h.insert(Vec3(1, 2, 3), 42));
    assert(!h.insert(Vec3(1, 2, 3), 99));
    assert(h.find(Vec3(1, 2, 3)) == 42);
    assert(h.count == 1);
}

unittest {
    // Remove and re-insert
    auto h = new OpenAddressingHash(10);
    h.insert(Vec3(1, 2, 3), 10);
    h.insert(Vec3(4, 5, 6), 20);
    assert(h.count == 2);

    assert(h.remove(Vec3(1, 2, 3)));
    assert(h.count == 1);
    assert(h.find(Vec3(1, 2, 3)) == -1);
    assert(h.find(Vec3(4, 5, 6)) == 20);  // unaffected

    // Re-insert at same position with different ID
    assert(h.insert(Vec3(1, 2, 3), 30));
    assert(h.find(Vec3(1, 2, 3)) == 30);
    assert(h.count == 2);
}

unittest {
    // Find works after tombstone in probe chain
    auto h = new OpenAddressingHash(4);  // small table (16 slots)
    // Insert several entries that might collide
    foreach (i; 0 .. 10)
        h.insert(Vec3(i * 0.1234, i * 0.5678, i * 0.9012), i);
    assert(h.count == 10);

    // Remove one in the middle
    h.remove(Vec3(3 * 0.1234, 3 * 0.5678, 3 * 0.9012));
    assert(h.count == 9);

    // Entries after the tombstone should still be findable
    foreach (i; 0 .. 10) {
        if (i == 3) {
            assert(h.find(Vec3(i * 0.1234, i * 0.5678, i * 0.9012)) == -1);
        } else {
            assert(h.find(Vec3(i * 0.1234, i * 0.5678, i * 0.9012)) == i);
        }
    }
}

unittest {
    // Many insertions don't crash (load factor test)
    auto h = new OpenAddressingHash(12);  // 4096 slots
    int inserted = 0;
    foreach (i; 0 .. 2000) {
        if (h.insert(Vec3(i * 0.001, i * 0.002, i * 0.003), i))
            inserted++;
    }
    assert(inserted == 2000);
    assert(h.count == 2000);

    // Verify all findable
    foreach (i; 0 .. 2000)
        assert(h.find(Vec3(i * 0.001, i * 0.002, i * 0.003)) == i);
}

unittest {
    // Not-found returns -1 for positions never inserted
    auto h = new OpenAddressingHash(10);
    h.insert(Vec3(0, 0, 0), 0);
    assert(h.find(Vec3(1, 0, 0)) == -1);
    assert(h.find(Vec3(0, 1, 0)) == -1);
    assert(h.find(Vec3(0.0001, 0, 0)) == -1);  // different key at tol=1e-7
}

unittest {
    // Positions within tolerance map to same key
    auto h = new OpenAddressingHash(10);
    h.insert(Vec3(1.0, 2.0, 3.0), 7);
    // 1e-8 offset should round to same key (tol = 1e-7)
    assert(h.find(Vec3(1.0 + 1e-8, 2.0 - 1e-8, 3.0 + 1e-8)) == 7);
}
