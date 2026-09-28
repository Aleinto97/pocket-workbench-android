plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.pocketworkbench.npubench"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.pocketworkbench.npubench"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += "arm64-v8a" }
    }
    // The HTP skel is a shared library, so it must be extracted and installed
    // by the package manager; extractNativeLibs=false would leave it compressed.
    packaging { jniLibs.useLegacyPackaging = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildTypes {
        getByName("release") { isMinifyEnabled = false }
    }
    sourceSets["main"].assets.srcDir(layout.projectDirectory.dir("src/main/assets"))
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
}
