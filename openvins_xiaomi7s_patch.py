from pathlib import Path

# arm64 only + do not require a USB-connected device during CI build
p = Path('app/build.gradle')
s = p.read_text()
s = s.replace("abiFilters 'x86', 'x86_64', 'arm64-v8a' // 'armeabi-v7a',", "abiFilters 'arm64-v8a'")
s = s.replace("task syncDeviceFiles {\n", "task syncDeviceFiles {\n    onlyIf { System.getenv('CI') != 'true' }\n")
p.write_text(s)

# Bundle upstream bootstrap calibration so APK can start without adb-pushing files.
assets = Path('app/src/main/assets/config')
assets.mkdir(parents=True, exist_ok=True)
for src in Path('app_device/config').glob('*.yaml'):
    (assets / src.name).write_bytes(src.read_bytes())

Path('app/src/main/res/values/strings.xml').write_text(
    '<resources>\n    <string name="app_name">VIO 坐标记录器</string>\n</resources>\n'
)

# Live position overlay.
layout = Path('app/src/main/res/layout/activity_main.xml')
x = layout.read_text()
overlay = r'''

    <LinearLayout
        android:id="@+id/coordinate_panel"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_alignParentTop="true"
        android:orientation="vertical"
        android:padding="12dp"
        android:background="#AA000000">

        <TextView
            android:id="@+id/coord_status"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="VIO：等待初始化（7S Pro 未精标定前仅用于测试）"
            android:textColor="#FFFFFF"
            android:textSize="16sp" />

        <TextView
            android:id="@+id/coord_x"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="X = +0.000 m"
            android:textColor="#FFFFFF"
            android:textSize="24sp" />

        <TextView
            android:id="@+id/coord_y"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Y = +0.000 m"
            android:textColor="#FFFFFF"
            android:textSize="24sp" />

        <TextView
            android:id="@+id/coord_z"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Z = +0.000 m"
            android:textColor="#FFFFFF"
            android:textSize="24sp" />

        <Button
            android:id="@+id/record_position"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="记录当前位置" />
    </LinearLayout>
'''
x = x.replace('\n</RelativeLayout>', overlay + '\n</RelativeLayout>')
layout.write_text(x)

main = Path('app/src/main/java/com/openvins/android/MainActivity.kt')
k = main.read_text()
k = k.replace(
    'import android.widget.Toast\n',
    'import android.widget.Toast\nimport android.widget.TextView\nimport android.widget.Button\nimport android.os.SystemClock\nimport java.io.FileWriter\nimport java.text.SimpleDateFormat\nimport java.util.Date\nimport java.util.Locale\n'
)
k = k.replace(
    '    private var trajectoryView: Trajectory3DView? = null\n',
    '''    private var trajectoryView: Trajectory3DView? = null
    private var coordX: TextView? = null
    private var coordY: TextView? = null
    private var coordZ: TextView? = null
    private var coordStatus: TextView? = null
    private var coordinateCsvFile: File? = null
    private var coordinateIndex: Int = 0
'''
)
k = k.replace(
    '        trajectoryView?.forceRender()\n',
    '''        trajectoryView?.forceRender()
        coordX = findViewById(R.id.coord_x)
        coordY = findViewById(R.id.coord_y)
        coordZ = findViewById(R.id.coord_z)
        coordStatus = findViewById(R.id.coord_status)
        findViewById<Button>(R.id.record_position).setOnClickListener { recordCurrentPosition() }
'''
)
k = k.replace(
    '''        val appRecordFolder =
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
                .toString() + "/openvins/"
''',
    '''        val appRecordFolder =
            (getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)?.toString() ?: appPrivateFolderRoot) + "/openvins/"
'''
)
k = k.replace(
    '        File(privateConfigDir).mkdirs()\n',
    '''        File(privateConfigDir).mkdirs()
        // Bootstrap only: replace with Xiaomi 7S Pro camera-IMU calibration for research use.
        for (name in listOf("estimator_config.yaml", "kalibr_imu_chain.yaml", "kalibr_imucam_chain.yaml")) {
            val dst = File(privateConfigDir, name)
            if (!dst.exists()) {
                try {
                    assets.open("config/$name").use { input -> dst.outputStream().use { output -> input.copyTo(output) } }
                } catch (e: Exception) {
                    Log.e(TAG, "Unable to copy bootstrap config $name", e)
                }
            }
        }
'''
)
k = k.replace(
    '''        if (!getCurrentPoseJNI(currentPos, currentQuat)) {
            return // System not initialized
        }
''',
    '''        if (!getCurrentPoseJNI(currentPos, currentQuat)) {
            coordStatus?.text = "VIO：等待初始化，请按 ▶ 并缓慢移动平板"
            return // System not initialized
        }
        coordStatus?.text = "VIO：TRACKING（未精标定）"
        coordX?.text = String.format(Locale.US, "X = %+.3f m", currentPos[0])
        coordY?.text = String.format(Locale.US, "Y = %+.3f m", currentPos[1])
        coordZ?.text = String.format(Locale.US, "Z = %+.3f m", currentPos[2])
'''
)
record_fn = '''    private fun recordCurrentPosition() {
        val p = DoubleArray(3)
        val q = DoubleArray(4)
        if (!getCurrentPoseJNI(p, q)) {
            Toast.makeText(this, "VIO 尚未初始化，先按 ▶ 并移动平板", Toast.LENGTH_SHORT).show()
            return
        }
        if (coordinateCsvFile == null) {
            val dir = File(recordFolder)
            dir.mkdirs()
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            coordinateCsvFile = File(dir, "coordinates_$stamp.csv")
            coordinateCsvFile!!.writeText("index,wall_time_ms,elapsed_realtime_ns,x_m,y_m,z_m,qw,qx,qy,qz\\n")
        }
        coordinateIndex += 1
        FileWriter(coordinateCsvFile!!, true).use { w ->
            w.append(String.format(Locale.US,
                "%d,%d,%d,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f\\n",
                coordinateIndex, System.currentTimeMillis(), SystemClock.elapsedRealtimeNanos(),
                p[0], p[1], p[2], q[0], q[1], q[2], q[3]))
        }
        Toast.makeText(this,
            String.format(Locale.US, "已记录 #%d  (%.3f, %.3f, %.3f) m", coordinateIndex, p[0], p[1], p[2]),
            Toast.LENGTH_SHORT).show()
    }

'''
k = k.replace('    companion object {\n', record_fn + '    companion object {\n')
main.write_text(k)
