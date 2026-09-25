plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}
android {
    namespace = "com.pocketworkbench.app"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.pocketworkbench.app"
        minSdk = 29
        targetSdk = 35
        versionCode = 5
        versionName = "0.2.4"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake { cppFlags += "-std=c++17"; arguments += listOf("-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF", "-DGGML_LLAMAFILE=OFF", "-DLLAMA_OPENSSL=OFF", "-DGGML_VULKAN=${if (providers.gradleProperty("gpu").orNull == "true") "ON" else "OFF"}") } }
    }
    buildFeatures { compose = true }
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
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    packaging { jniLibs.useLegacyPackaging = true }
}
dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
