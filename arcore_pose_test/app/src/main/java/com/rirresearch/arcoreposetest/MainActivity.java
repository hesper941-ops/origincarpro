package com.rirresearch.arcoreposetest;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.TextView;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingFailureReason;
import com.google.ar.core.TrackingState;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;

import java.util.Locale;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class MainActivity extends Activity implements GLSurfaceView.Renderer {
    private static final int CAMERA_PERMISSION = 1001;

    private GLSurfaceView glView;
    private TextView statusView;
    private Button originButton;
    private Session session;
    private boolean sessionResumed = false;
    private boolean installRequested = false;
    private volatile Pose latestValidPose = null;
    private volatile Pose originPose = null;
    private int cameraTextureId = -1;
    private int surfaceWidth = 1;
    private int surfaceHeight = 1;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private long lastUiUpdateNs = 0L;
    private String availabilityText = "checking...";
    private String sessionText = "NOT STARTED";
    private String lastError = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION);
        } else {
            checkAvailabilityAndStart();
        }
    }

    private void buildUi() {
        FrameLayout root = new FrameLayout(this);

        glView = new GLSurfaceView(this);
        glView.setEGLContextClientVersion(2);
        glView.setPreserveEGLContextOnPause(true);
        glView.setRenderer(this);
        glView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
        root.addView(glView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));

        statusView = new TextView(this);
        statusView.setTextSize(16f);
        statusView.setTextColor(0xFF111111);
        statusView.setBackgroundColor(0xEEFFFFFF);
        statusView.setPadding(24, 24, 24, 24);
        statusView.setText("ARCore Pose Test\nStarting...");
        FrameLayout.LayoutParams statusLp = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        statusLp.gravity = Gravity.TOP;
        root.addView(statusView, statusLp);

        originButton = new Button(this);
        originButton.setText("SET ORIGIN");
        originButton.setEnabled(false);
        originButton.setOnClickListener(v -> {
            Pose p = latestValidPose;
            if (p != null) originPose = p;
        });
        FrameLayout.LayoutParams buttonLp = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        buttonLp.gravity = Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL;
        buttonLp.bottomMargin = 48;
        root.addView(originButton, buttonLp);

        setContentView(root);
    }

    private void checkAvailabilityAndStart() {
        ArCoreApk.Availability availability = ArCoreApk.getInstance().checkAvailability(this);
        availabilityText = availability.name();
        refreshStaticUi();

        if (availability.isTransient()) {
            mainHandler.postDelayed(this::checkAvailabilityAndStart, 300);
            return;
        }
        if (!availability.isSupported()) {
            sessionText = "NOT STARTED";
            lastError = "ARCore runtime verdict: " + availability.name();
            refreshStaticUi();
            return;
        }
        createSession();
    }

    private void createSession() {
        if (session != null) return;
        try {
            ArCoreApk.InstallStatus installStatus =
                    ArCoreApk.getInstance().requestInstall(this, !installRequested);
            if (installStatus == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                installRequested = true;
                sessionText = "WAITING FOR AR SERVICE";
                refreshStaticUi();
                return;
            }

            session = new Session(this);
            Config config = new Config(session);
            config.setUpdateMode(Config.UpdateMode.LATEST_CAMERA_IMAGE);
            session.configure(config);
            sessionText = "CREATED";
            lastError = "";

            if (cameraTextureId > 0) {
                glView.queueEvent(() -> {
                    Session s = session;
                    if (s != null) s.setCameraTextureName(cameraTextureId);
                });
            }
            resumeSession();
            refreshStaticUi();
        } catch (UnavailableArcoreNotInstalledException e) {
            fail("UnavailableArcoreNotInstalledException", e);
        } catch (UnavailableApkTooOldException e) {
            fail("UnavailableApkTooOldException", e);
        } catch (UnavailableSdkTooOldException e) {
            fail("UnavailableSdkTooOldException", e);
        } catch (UnavailableDeviceNotCompatibleException e) {
            fail("UnavailableDeviceNotCompatibleException", e);
        } catch (Exception e) {
            fail(e.getClass().getSimpleName(), e);
        }
    }

    private void resumeSession() {
        if (session == null || sessionResumed) return;
        try {
            session.resume();
            sessionResumed = true;
            sessionText = "RUNNING";
            lastError = "";
        } catch (CameraNotAvailableException e) {
            sessionText = "FAILED";
            lastError = "CameraNotAvailableException: " + safeMsg(e);
        } catch (Exception e) {
            sessionText = "FAILED";
            lastError = e.getClass().getSimpleName() + ": " + safeMsg(e);
        }
    }

    private void fail(String label, Exception e) {
        sessionText = "FAILED";
        lastError = label + ": " + safeMsg(e);
        refreshStaticUi();
    }

    private String safeMsg(Exception e) {
        return e.getMessage() == null ? "(no message)" : e.getMessage();
    }

    private void refreshStaticUi() {
        runOnUiThread(() -> statusView.setText(
                "ARCore Pose Test (standalone)\n" +
                "Package: com.rirresearch.arcoreposetest\n\n" +
                "Availability: " + availabilityText + "\n" +
                "Session: " + sessionText + "\n" +
                (lastError.isEmpty() ? "" : "Error: " + lastError + "\n") +
                "\nWaiting for tracking..."));
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (glView != null) glView.onResume();
        if (session == null) {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                checkAvailabilityAndStart();
            }
        } else {
            resumeSession();
        }
    }

    @Override
    protected void onPause() {
        if (glView != null) glView.onPause();
        if (session != null && sessionResumed) {
            session.pause();
            sessionResumed = false;
        }
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        if (session != null) {
            session.close();
            session = null;
            sessionResumed = false;
        }
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                checkAvailabilityAndStart();
            } else {
                sessionText = "NOT STARTED";
                lastError = "Camera permission DENIED";
                refreshStaticUi();
            }
        }
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES20.glClearColor(0f, 0f, 0f, 1f);
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        cameraTextureId = textures[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        Session s = session;
        if (s != null) s.setCameraTextureName(cameraTextureId);
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        surfaceWidth = Math.max(width, 1);
        surfaceHeight = Math.max(height, 1);
        GLES20.glViewport(0, 0, surfaceWidth, surfaceHeight);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        Session s = session;
        if (s == null || !sessionResumed || cameraTextureId <= 0) return;

        try {
            int rotation = getWindowManager().getDefaultDisplay().getRotation();
            s.setDisplayGeometry(rotation, surfaceWidth, surfaceHeight);
            Frame frame = s.update();
            Camera camera = frame.getCamera();
            TrackingState trackingState = camera.getTrackingState();
            TrackingFailureReason failureReason = camera.getTrackingFailureReason();
            Pose worldPose = camera.getPose();
            if (trackingState == TrackingState.TRACKING) latestValidPose = worldPose;

            long now = System.nanoTime();
            if (now - lastUiUpdateNs < 100_000_000L) return;
            lastUiUpdateNs = now;

            Pose relative = null;
            Pose origin = originPose;
            if (origin != null && trackingState == TrackingState.TRACKING) {
                relative = origin.inverse().compose(worldPose);
            }

            final Pose relFinal = relative;
            final long frameTimestamp = frame.getTimestamp();
            final TrackingState tsFinal = trackingState;
            final TrackingFailureReason frFinal = failureReason;
            final Pose wpFinal = worldPose;
            runOnUiThread(() -> updatePoseUi(frameTimestamp, tsFinal, frFinal, wpFinal, relFinal));
        } catch (Exception e) {
            final String error = e.getClass().getSimpleName() + ": " +
                    (e.getMessage() == null ? "(no message)" : e.getMessage());
            runOnUiThread(() -> {
                lastError = error;
                sessionText = "UPDATE ERROR";
                refreshStaticUi();
            });
        }
    }

    private void updatePoseUi(long frameTimestamp,
                              TrackingState trackingState,
                              TrackingFailureReason failureReason,
                              Pose worldPose,
                              Pose relativePose) {
        boolean valid = trackingState == TrackingState.TRACKING;
        originButton.setEnabled(valid);

        StringBuilder sb = new StringBuilder();
        sb.append("ARCore Pose Test (standalone)\n");
        sb.append("Package: com.rirresearch.arcoreposetest\n\n");
        sb.append("Availability: ").append(availabilityText).append("\n");
        sb.append("Session: ").append(sessionText).append("\n");
        sb.append("Tracking: ").append(trackingState.name()).append("\n");
        sb.append("Failure reason: ").append(failureReason.name()).append("\n");
        sb.append("Frame timestamp: ").append(frameTimestamp).append(" ns\n\n");

        if (valid) {
            sb.append(String.format(Locale.US,
                    "WORLD POSE (m)\nX: %.4f\nY: %.4f\nZ: %.4f\n",
                    worldPose.tx(), worldPose.ty(), worldPose.tz()));
            sb.append(String.format(Locale.US,
                    "Quaternion\nqx: %.6f\nqy: %.6f\nqz: %.6f\nqw: %.6f\n\n",
                    worldPose.qx(), worldPose.qy(), worldPose.qz(), worldPose.qw()));
            if (relativePose == null) {
                sb.append("RELATIVE POSE\nOrigin not set.\n");
            } else {
                sb.append(String.format(Locale.US,
                        "RELATIVE POSE (m)\nX: %.4f\nY: %.4f\nZ: %.4f\n",
                        relativePose.tx(), relativePose.ty(), relativePose.tz()));
            }
        } else {
            sb.append("POSE INVALID while not TRACKING.\n");
        }

        if (!lastError.isEmpty()) sb.append("\nError: ").append(lastError).append("\n");
        statusView.setText(sb.toString());
    }
}
