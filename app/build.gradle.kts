plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}
android {
    namespace = "com.pocketworkbench.app"
    compileSdk = 35
    ndkVersion = "27.2.12479018"
    defaultConfig {
        applicationId = "com.pocketworkbench.app"
        minSdk = 29
        targetSdk = 35
        versionCode = 21
        versionName = "0.3.9-harness"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake {
            cppFlags += "-std=c++17"
            arguments += listOf("-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF")
        } }
    }
    buildFeatures { compose = true; aidl = true }
    // Stable signing key committed in the private repo: every CI build is signed
    // identically so new APKs install as UPDATES over previous ones (model files
    // in filesDir survive). Without this, each CI runner generated a fresh debug
    // key and users had to uninstall (losing downloaded models) for every update.
    signingConfigs {
        create("stable") {
            storeFile = file("pocketworkbench.keystore")
            storePassword = "pocketworkbench2026"
            keyAlias = "pocketworkbench"
            keyPassword = "pocketworkbench2026"
        }
    }
    buildTypes {
        getByName("debug") { signingConfig = signingConfigs.getByName("stable") }
        getByName("release") {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("stable")
        }
    }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    if (providers.gradleProperty("usePrebuiltSpeech").orNull != "true") {
        externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    }
    sourceSets["main"].jniLibs.srcDir(layout.buildDirectory.dir("rustjni"))
    // The NPU model service needs the same GenieX runtime npubench measures,
    // so the shipped app can answer model requests without a second APK.
    sourceSets["main"].jniLibs.srcDir(layout.buildDirectory.dir("geniexjni"))
    // Static proot + curl for the optional Linux module and the Android shell.
    // Small (~10 MiB) and always shipped; the Debian rootfs stays on-demand.
    sourceSets["main"].jniLibs.srcDir(layout.buildDirectory.dir("linuxjni"))
    if (providers.gradleProperty("usePrebuiltSpeech").orNull == "true") {
        sourceSets["main"].jniLibs.srcDir(layout.buildDirectory.dir("prebuiltjni"))
    }
    androidResources { noCompress += "bin" }
    packaging { jniLibs.useLegacyPackaging = true }
}

// Pure-Rust inference engine (rust/pocketinfer): cross-compiled by
// scripts/build-rust.sh into app/build/rustjni/arm64-v8a/libpocketinfer.so.
// inputs.dir ensures a Rust change actually rebuilds the .so instead of
// staying UP-TO-DATE with a stale library that lacks the new JNI symbols.
val cargoBuild = tasks.register<Exec>("cargoBuildRust") {
    workingDir = rootProject.projectDir
    commandLine("bash", "scripts/build-rust.sh")
    inputs.dir(rootProject.projectDir.resolve("rust/pocketinfer/src"))
        .withPropertyName("rustSources")
        .withPathSensitivity(PathSensitivity.RELATIVE)
    inputs.file(rootProject.projectDir.resolve("rust/pocketinfer/Cargo.toml"))
        .withPropertyName("rustManifest")
        .withPathSensitivity(PathSensitivity.RELATIVE)
    inputs.file(rootProject.projectDir.resolve("scripts/build-rust.sh"))
        .withPropertyName("rustBuildScript")
        .withPathSensitivity(PathSensitivity.RELATIVE)
    outputs.dir(layout.buildDirectory.dir("rustjni"))
}
tasks.named("preBuild") { dependsOn(cargoBuild) }
dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.animation:animation")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    testImplementation("junit:junit:4.13.2")
}

tasks.matching { it.name == "mergeDebugJniLibFolders" || it.name == "mergeReleaseJniLibFolders" }.configureEach {
    doFirst {
        check(layout.buildDirectory.file("geniexjni/geniexjni.json").get().asFile.isFile) {
            "GenieX runtime missing. Run python3 scripts/package-geniex-runtime.py first."
        }
        check(layout.buildDirectory.file("linuxjni/linuxjni.json").get().asFile.isFile) {
            "Linux runtime missing. Run bash scripts/fetch-linux.sh, then python3 scripts/package-linux-runtime.py first."
        }
    }
}
