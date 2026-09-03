plugins {
    id("com.android.application")
}

android {
    namespace = "com.example.pptimageorderfix"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.pptimageorderfix"
        minSdk = 29
        targetSdk = 35
        versionCode = 3
        versionName = "1.2"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation("androidx.exifinterface:exifinterface:1.3.7")
}
