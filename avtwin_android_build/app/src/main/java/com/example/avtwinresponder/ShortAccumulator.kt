package com.example.avtwinresponder

class ShortAccumulator(initialCapacity: Int = 48000 * 8) {
    private var data = ShortArray(initialCapacity.coerceAtLeast(1024))
    var size: Int = 0
        private set

    fun append(src: ShortArray, length: Int) {
        require(length <= src.size)
        ensure(size + length)
        System.arraycopy(src, 0, data, size, length)
        size += length
    }

    operator fun get(index: Int): Short = data[index]

    fun copy(): ShortArray = data.copyOf(size)

    private fun ensure(required: Int) {
        if (required <= data.size) return
        var cap = data.size
        while (cap < required) cap *= 2
        data = data.copyOf(cap)
    }
}
