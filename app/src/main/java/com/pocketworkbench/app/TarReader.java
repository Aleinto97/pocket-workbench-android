package com.pocketworkbench.app;

import java.io.Closeable;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Minimal tar reader: ustar plus the GNU long-name extension. Deliberately
 * small and dependency-free — it exists so the app builds fully offline.
 *
 * <p>Supported entries: regular files, directories, symlinks, hard links.
 * GNU {@code L}/{@code K} long-name blocks rename the following entry. Pax
 * extended headers ({@code x}/{@code g}) are consumed and ignored. Device
 * nodes, fifos and sockets are reported and left for the caller to skip:
 * recreating them needs privileges nobody here has.
 *
 * <p>The caller owns policy (path traversal, name stripping, filtering);
 * this class only parses sizes and padding exactly, because one wrong skip
 * desynchronises everything after it.
 */
final class TarReader implements Closeable {
    static final class Entry {
        /** Raw path as stored in the archive, '/' separated. */
        String name = "";
        /** '0' file, '5' dir, '2' symlink, '1' hardlink, else the raw flag. */
        char type;
        /** Link target for symlinks and hard links. */
        String linkName = "";
        /** Payload bytes for files. */
        long size;
        /** Unix mode bits, 0 when absent. */
        int mode;
    }

    private static final int BLOCK = 512;

    private final InputStream in;
    private final byte[] block = new byte[BLOCK];
    private String pendingLongName;
    private String pendingLongLink;
    /** From a per-entry ('x') pax header: renames the very next entry. */
    private String pendingPaxPath;
    private String pendingPaxLink;
    /** From a global ('g') pax header: fallback for truncated names only. */
    private String globalPaxPath;
    private String globalPaxLink;
    private long remaining;
    private boolean closed;

    TarReader(InputStream in) {
        this.in = in;
    }

    /**
     * Reads the next entry header. Returns null at the end of the archive
     * (two zero blocks, or clean EOF right on a block boundary).
     */
    Entry next() throws IOException {
        remaining = 0;
        for (;;) {
            try {
                readBlock();
            } catch (EOFException e) {
                return null;
            }
            if (isZeroBlock()) {
                // The end marker is two zero blocks; tolerate a lone one.
                try {
                    readBlock();
                } catch (EOFException e) {
                    return null;
                }
                if (isZeroBlock()) {
                    return null;
                }
                // Otherwise fall through and parse the block just read.
            }
            Entry entry = parseHeader();
            if (entry == null) {
                continue; // pax header consumed; read the real one next
            }
            if (entry.type == 'L' || entry.type == 'K') {
                String text = readTextPayload(entry.size);
                if (entry.type == 'L') {
                    pendingLongName = text;
                } else {
                    pendingLongLink = text;
                }
                skipPadding(entry.size);
                continue;
            }
            if (pendingLongName != null) {
                entry.name = pendingLongName;
                pendingLongName = null;
                nameTruncated = false; // the real name arrived out of band
            }
            if (pendingLongLink != null && (entry.type == '2' || entry.type == '1')) {
                entry.linkName = pendingLongLink;
                pendingLongLink = null;
            }
            if (pendingPaxPath != null) {
                entry.name = pendingPaxPath;
                pendingPaxPath = null;
                nameTruncated = false;
            }
            if (pendingPaxLink != null && (entry.type == '2' || entry.type == '1')) {
                entry.linkName = pendingPaxLink;
                pendingPaxLink = null;
            }
            if (globalPaxPath != null && nameTruncated) {
                entry.name = globalPaxPath;
            }
            if (globalPaxLink != null && (entry.type == '2' || entry.type == '1') && entry.linkName.length() >= 100) {
                entry.linkName = globalPaxLink;
            }
            lastSize = entry.type == '0' ? entry.size : 0;
            remaining = lastSize;
            if (entry.type != '0') {
                skipPadding(entry.size);
            }
            return entry;
        }
    }

    /**
     * Streams the current file entry's payload. Must be called exactly once
     * per file entry before {@link #next()}; consumes trailing padding.
     */
    void copyTo(OutputStream out) throws IOException {
        byte[] buf = new byte[32 * 1024];
        long left = remaining;
        while (left > 0) {
            int want = (int) Math.min(buf.length, left);
            int got = in.read(buf, 0, want);
            if (got < 0) {
                throw new IOException("tar ended mid-file");
            }
            out.write(buf, 0, got);
            left -= got;
        }
        remaining = 0;
        skipPadding(lastSize);
    }

    private long lastSize;

    private boolean nameTruncated;

    /** Parses {@link #block}; returns null for pax headers (payload skipped). */
    private Entry parseHeader() throws IOException {
        String magic = ascii(257, 6).trim();
        if (!magic.startsWith("ustar") && !magic.isEmpty()) {
            throw new IOException("not a ustar tar entry (bad magic)");
        }
        Entry entry = new Entry();
        String name = ascii(0, 100);
        nameTruncated = name.length() >= 100;
        String prefix = ascii(345, 155);
        entry.name = prefix.isEmpty() ? name : prefix + '/' + name;
        entry.mode = parseOctal(100, 8);
        entry.size = parseOctalLong(124, 12);
        char flag = (char) block[156];
        entry.type = flag == 0 ? '0' : flag;
        entry.linkName = ascii(157, 100);
        if (entry.type == 'x' || entry.type == 'g') {
            // Pax extended header: key=value records rename the next entry
            // ('x') or set defaults ('g'). Only path/linkpath matter here;
            // atime/ctime and friends are ignored.
            String pax = readTextPayloadCapped(entry.size, 1024 * 1024);
            skipPadding(entry.size);
            String path = paxValue(pax, "path");
            String linkpath = paxValue(pax, "linkpath");
            if (entry.type == 'x') {
                if (path != null) {
                    pendingPaxPath = path;
                }
                if (linkpath != null) {
                    pendingPaxLink = linkpath;
                }
            } else {
                if (path != null) {
                    globalPaxPath = path;
                }
                if (linkpath != null) {
                    globalPaxLink = linkpath;
                }
            }
            return null;
        }
        return entry;
    }

    /**
     * Reads a pax record block: lines of "<length> key=value", where length
     * covers the whole line. Returns the value for {@code key}, or null.
     */
    private static String paxValue(String pax, String key) {
        String want = key + "=";
        int from = 0;
        while (from < pax.length()) {
            int end = pax.indexOf('\n', from);
            if (end < 0) {
                end = pax.length();
            }
            String line = pax.substring(from, end);
            int space = line.indexOf(' ');
            if (space > 0 && line.startsWith(want, space + 1)) {
                return line.substring(space + 1 + want.length());
            }
            from = end + 1;
        }
        return null;
    }

    private String readTextPayload(long size) throws IOException {
        return readTextPayloadCapped(size, 16 * 1024 * 1024);
    }

    private String readTextPayloadCapped(long size, long cap) throws IOException {
        if (size > cap) {
            throw new IOException("implausible text block");
        }
        byte[] data = new byte[(int) size];
        readFully(data, 0, data.length);
        int end = data.length;
        while (end > 0 && data[end - 1] == 0) {
            end--;
        }
        return new String(data, 0, end, StandardCharsets.UTF_8);
    }

    private void skipPadding(long size) throws IOException {
        long pad = (BLOCK - (size % BLOCK)) % BLOCK;
        while (pad > 0) {
            long skipped = in.skip(pad);
            if (skipped <= 0) {
                // skip() may legally return 0: fall back to a read.
                int b = in.read();
                if (b < 0) {
                    throw new IOException("tar ended in padding");
                }
                pad--;
            } else {
                pad -= skipped;
            }
        }
    }

    private boolean isZeroBlock() {
        for (byte b : block) {
            if (b != 0) {
                return false;
            }
        }
        return true;
    }

    private void readBlock() throws IOException {
        readFully(block, 0, BLOCK);
    }

    private void readFully(byte[] buf, int off, int len) throws IOException {
        int got = 0;
        while (got < len) {
            int n = in.read(buf, off + got, len - got);
            if (n < 0) {
                throw new EOFException("unexpected end of tar stream");
            }
            got += n;
        }
    }

    private String ascii(int off, int len) {
        int end = off;
        while (end < off + len && block[end] != 0) {
            end++;
        }
        return new String(block, off, end - off, StandardCharsets.UTF_8);
    }

    private int parseOctal(int off, int len) {
        return (int) parseOctalLong(off, len);
    }

    private long parseOctalLong(int off, int len) {
        // Base-256 (binary) GNU extension: high bit set means the rest is
        // big-endian binary, used for huge files and large ids.
        if (block[off] < 0) {
            long value = 0;
            for (int i = off + 1; i < off + len; i++) {
                value = (value << 8) | (block[i] & 0xFF);
            }
            return value;
        }
        long value = 0;
        for (int i = off; i < off + len; i++) {
            byte b = block[i];
            if (b == 0 || b == ' ') {
                break;
            }
            if (b < '0' || b > '7') {
                throw new NumberFormatException("bad octal in tar header");
            }
            value = (value << 3) | (b - '0');
        }
        return value;
    }

    @Override
    public void close() throws IOException {
        if (!closed) {
            closed = true;
            in.close();
        }
    }
}
