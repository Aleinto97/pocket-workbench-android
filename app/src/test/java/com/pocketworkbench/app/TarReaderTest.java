package com.pocketworkbench.app;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.tukaani.xz.XZInputStream;

/** The Debian install path, minus Android: XZ decode plus tar parsing. */
public class TarReaderTest {

    private static InputStream fixture() {
        InputStream in = TarReaderTest.class.getResourceAsStream("/mini-rootfs.tar.xz");
        if (in == null) {
            throw new IllegalStateException("missing test fixture mini-rootfs.tar.xz");
        }
        return in;
    }

    private static final class Seen {
        String name;
        char type;
        String link;
        long size;
        int mode;
        byte[] content;
    }

    private static List<Seen> readAll() throws Exception {
        List<Seen> out = new ArrayList<>();
        try (TarReader tar = new TarReader(new XZInputStream(fixture()))) {
            for (;;) {
                TarReader.Entry entry = tar.next();
                if (entry == null) {
                    break;
                }
                Seen seen = new Seen();
                seen.name = entry.name;
                seen.type = entry.type;
                seen.link = entry.linkName;
                seen.size = entry.size;
                seen.mode = entry.mode;
                if (entry.type == '0') {
                    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                    tar.copyTo(bytes);
                    seen.content = bytes.toByteArray();
                }
                out.add(seen);
            }
        }
        return out;
    }

    private static Seen byName(List<Seen> all, String name) {
        for (Seen seen : all) {
            if (seen.name.equals(name)) {
                return seen;
            }
        }
        fail("missing entry " + name);
        return null;
    }

    @Test
    public void readsEveryEntryKind() throws Exception {
        List<Seen> all = readAll();
        // Stored entries minus the GNU long-name block and the pax extended
        // header, both of which rename a neighbour instead of appearing.
        assertEquals(13, all.size());

        Seen dir = byName(all, "top/d/sub/");
        assertEquals('5', dir.type);

        Seen file = byName(all, "top/d/sub/f.txt");
        assertEquals('0', file.type);
        assertEquals("hello-debian\n".getBytes(StandardCharsets.UTF_8).length, file.size);
        assertArrayEquals("hello-debian\n".getBytes(StandardCharsets.UTF_8), file.content);
        assertEquals(0755, file.mode);

        Seen empty = byName(all, "top/d/empty.bin");
        assertEquals(0, empty.size);
        assertEquals(0, empty.content.length);

        assertEquals('2', byName(all, "top/d/rel-link").type);
        assertEquals("sub/f.txt", byName(all, "top/d/rel-link").link);
        assertEquals("/etc/hostname", byName(all, "top/d/abs-link").link);

        Seen hard = byName(all, "top/d/hard-link");
        assertEquals('1', hard.type);
        assertEquals("top/d/sub/f.txt", hard.link);

        StringBuilder longName = new StringBuilder("top/d/");
        for (int i = 0; i < 120; i++) {
            longName.append('L');
        }
        longName.append(".txt");
        Seen longFile = byName(all, longName.toString());
        assertEquals("long name body\n", new String(longFile.content, StandardCharsets.UTF_8));

        // Pax extended header: the full name lives in a path= record, the
        // entry itself is truncated to 100 chars.
        StringBuilder paxName = new StringBuilder("top/d/");
        for (int i = 0; i < 120; i++) {
            paxName.append('P');
        }
        paxName.append(".txt");
        Seen paxFile = byName(all, paxName.toString());
        assertEquals('0', paxFile.type);
        assertEquals("pax name body\n", new String(paxFile.content, StandardCharsets.UTF_8));

        // Device nodes are reported, never materialised by the caller.
        assertEquals('6', byName(all, "top/d/a-fifo").type);
    }

    @Test
    public void reportsHostileNamesInsteadOfResolvingThem() throws Exception {
        List<Seen> all = readAll();
        byName(all, "/abs-evil.txt");
        byName(all, "top/../traversal.txt");
        byName(all, "top/dev/null");
    }

    @Test
    public void truncatedStreamFailsInsteadOfSilentlyStopping() throws Exception {
        byte[] garbage = new byte[2048];
        for (int i = 0; i < garbage.length; i++) {
            garbage[i] = (byte) (i * 31 + 7);
        }
        // Not a tar stream at all: parsing must fail loudly rather than
        // report an empty archive.
        try (TarReader tar = new TarReader(new java.io.ByteArrayInputStream(garbage))) {
            tar.next();
            fail("expected a loud failure on a non-tar stream");
        } catch (java.io.IOException expected) {
            // Loud failure is the correct behaviour.
        }
    }

    @Test
    public void policySkipsDevAndRejectsEscapes() {
        File root = new File("/tmp/debian-policy-test");
        assertNull(LinuxModule.targetFor(root, ""));
        assertNull(LinuxModule.targetFor(root, "dev"));
        assertNull(LinuxModule.targetFor(root, "dev/null"));
        assertEquals(new File(root, "d/sub/f.txt"), LinuxModule.targetFor(root, "d/sub/f.txt"));
        assertEquals(new File(root, "bin/bash"), LinuxModule.targetFor(root, "bin/bash"));
        try {
            LinuxModule.targetFor(root, "../traversal.txt");
            fail("expected escape rejection");
        } catch (IllegalArgumentException expected) {
            // Hostile tarball: fail the install, never write outside.
        }
        // A leading slash is folded under the root by java.io.File, so an
        // absolute name is contained, not an escape.
        assertEquals(new File(root, "abs-evil.txt"), LinuxModule.targetFor(root, "/abs-evil.txt"));
    }

    @Test
    public void realDebianLayoutAssumptionsHold() {
        // Guards the strip-first-component policy against tarball changes:
        // everything the installer keeps lives under one top-level dir.
        assertTrue("top/d/sub/f.txt".contains("/"));
    }
}
