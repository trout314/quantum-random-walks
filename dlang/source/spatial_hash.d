/**
 * spatial_hash.d — Spatial hash table for 3D site lookup.
 *
 * Maps 3D positions (discretized to integer keys) to site IDs.
 * Implemented as a struct for zero-overhead inlining on the hot path.
 * To swap implementations, change the SiteHash alias below.
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

/// Open-addressing hash table with linear probing and tombstone support.
struct OpenAddressingHash {
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

    static OpenAddressingHash create(int bits) {
        assert(bits > 0 && bits <= 30);
        OpenAddressingHash h;
        uint size = 1u << bits;
        h.mask = size - 1;
        h.table = new Entry[size];
        h._count = 0;
        return h;
    }

    private static uint hash(long kx, long ky, long kz) {
        return cast(uint)(
            (cast(ulong)(kx * 73856093L)
           ^ cast(ulong)(ky * 19349663L)
           ^ cast(ulong)(kz * 83492791L))
        );
    }

    int find(Vec3 pos) const {
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

    @property int count() const { return _count; }
}

/// Default hash table type. Change this alias to swap implementations.
alias SiteHash = OpenAddressingHash;

// ---- D unit tests ----

unittest {
    // Basic insert and find
    auto h = OpenAddressingHash.create(10);
    assert(h.find(Vec3(1, 2, 3)) == -1);
    assert(h.insert(Vec3(1, 2, 3), 42));
    assert(h.find(Vec3(1, 2, 3)) == 42);
    assert(h.count == 1);
}

unittest {
    // Duplicate insert rejected
    auto h = OpenAddressingHash.create(10);
    assert(h.insert(Vec3(1, 2, 3), 42));
    assert(!h.insert(Vec3(1, 2, 3), 99));
    assert(h.find(Vec3(1, 2, 3)) == 42);
    assert(h.count == 1);
}

unittest {
    // Remove and re-insert
    auto h = OpenAddressingHash.create(10);
    h.insert(Vec3(1, 2, 3), 10);
    h.insert(Vec3(4, 5, 6), 20);
    assert(h.count == 2);

    assert(h.remove(Vec3(1, 2, 3)));
    assert(h.count == 1);
    assert(h.find(Vec3(1, 2, 3)) == -1);
    assert(h.find(Vec3(4, 5, 6)) == 20);

    assert(h.insert(Vec3(1, 2, 3), 30));
    assert(h.find(Vec3(1, 2, 3)) == 30);
    assert(h.count == 2);
}

unittest {
    // Find works after tombstone in probe chain
    auto h = OpenAddressingHash.create(4);  // 16 slots
    foreach (i; 0 .. 10)
        h.insert(Vec3(i * 0.1234, i * 0.5678, i * 0.9012), i);
    assert(h.count == 10);

    h.remove(Vec3(3 * 0.1234, 3 * 0.5678, 3 * 0.9012));
    assert(h.count == 9);

    foreach (i; 0 .. 10) {
        if (i == 3)
            assert(h.find(Vec3(i * 0.1234, i * 0.5678, i * 0.9012)) == -1);
        else
            assert(h.find(Vec3(i * 0.1234, i * 0.5678, i * 0.9012)) == i);
    }
}

unittest {
    // Load factor test
    auto h = OpenAddressingHash.create(12);  // 4096 slots
    foreach (i; 0 .. 2000)
        assert(h.insert(Vec3(i * 0.001, i * 0.002, i * 0.003), i));
    assert(h.count == 2000);
    foreach (i; 0 .. 2000)
        assert(h.find(Vec3(i * 0.001, i * 0.002, i * 0.003)) == i);
}

unittest {
    // Positions within tolerance map to same key
    auto h = OpenAddressingHash.create(10);
    h.insert(Vec3(1.0, 2.0, 3.0), 7);
    assert(h.find(Vec3(1.0 + 1e-8, 2.0 - 1e-8, 3.0 + 1e-8)) == 7);
}
