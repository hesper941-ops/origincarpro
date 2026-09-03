package com.example.pptimageorderfix;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.exifinterface.media.ExifInterface;

import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TimeZone;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MainActivity extends Activity {
    private static final int REQ_FOLDER = 1001;

    private Uri selectedTreeUri;
    private TextView folderText;
    private TextView statusText;
    private Button startButton;
    private CheckBox zeroBasedBox;

    private static final Pattern PAGE_SUFFIX =
            Pattern.compile("_(\\d+)\\.(jpg|jpeg|png|webp)$", Pattern.CASE_INSENSITIVE);

    static class ImageItem {
        Uri uri;
        String name;
        String mime;
        String ext;
        int suffixNumber;
        int pageNumber;
        String batchKey;
        long batchId;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(20);
        root.setPadding(pad, pad, pad, pad);
        scroll.addView(root);

        TextView title = new TextView(this);
        title.setText("PPT 图片顺序修复 v1.3");
        title.setTextSize(26);
        title.setPadding(0, 0, 0, dp(12));
        root.addView(title);

        TextView desc = new TextView(this);
        desc.setText(
                "针对 Notein 再加强排序：所有页都放在同一天内，只相差 1 秒，避免被按日期/时间段拆组。\n\n" +
                "同时统一文件名、TITLE、DATE_TAKEN、DATE_ADDED、DATE_MODIFIED、EXIF 时间，并保持第一页最新。\n\n" +
                "不会修改或删除原文件。");
        desc.setTextSize(16);
        root.addView(desc);

        Button chooseButton = new Button(this);
        chooseButton.setText("① 选择 QQ 浏览器导出的图片文件夹");
        chooseButton.setOnClickListener(v -> chooseFolder());
        root.addView(chooseButton);

        folderText = new TextView(this);
        folderText.setText("尚未选择文件夹");
        folderText.setTextSize(14);
        folderText.setPadding(0, dp(8), 0, dp(8));
        root.addView(folderText);

        zeroBasedBox = new CheckBox(this);
        zeroBasedBox.setText("QQ 文件名从 0 开始编号（_87 = 第 88 页）");
        zeroBasedBox.setChecked(true);
        root.addView(zeroBasedBox);

        startButton = new Button(this);
        startButton.setText("② 生成 Notein 同日严格排序相册");
        startButton.setEnabled(false);
        startButton.setOnClickListener(v -> startFix());
        root.addView(startButton);

        statusText = new TextView(this);
        statusText.setText("等待操作");
        statusText.setTextSize(15);
        statusText.setPadding(0, dp(16), 0, dp(24));
        root.addView(statusText);

        setContentView(scroll);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void chooseFolder() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION |
                Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION |
                Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);
        startActivityForResult(intent, REQ_FOLDER);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQ_FOLDER && resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            if (uri == null) return;
            try {
                getContentResolver().takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
            } catch (Exception ignored) {}
            selectedTreeUri = uri;
            folderText.setText("已选择：\n" + uri);
            startButton.setEnabled(true);
            statusText.setText("可以开始整理");
        }
    }

    private void startFix() {
        if (selectedTreeUri == null) {
            Toast.makeText(this, "请先选择图片文件夹", Toast.LENGTH_SHORT).show();
            return;
        }

        startButton.setEnabled(false);
        statusText.setText("正在扫描图片…");

        new Thread(() -> {
            try {
                List<ImageItem> items = scanFolder(selectedTreeUri, zeroBasedBox.isChecked());
                if (items.isEmpty()) throw new Exception("没有找到 _数字.jpg/png/webp 格式的图片");

                Map<String, List<ImageItem>> batches = new LinkedHashMap<>();
                for (ImageItem item : items) {
                    batches.computeIfAbsent(item.batchKey, k -> new ArrayList<>()).add(item);
                }

                List<ImageItem> selected = null;
                long newestBatchId = Long.MIN_VALUE;
                for (List<ImageItem> batch : batches.values()) {
                    long id = batch.isEmpty() ? 0L : batch.get(0).batchId;
                    if (selected == null || id > newestBatchId ||
                            (id == newestBatchId && batch.size() > selected.size())) {
                        newestBatchId = id;
                        selected = batch;
                    }
                }
                if (selected == null || selected.isEmpty()) throw new Exception("没有可处理批次");

                TreeMap<Integer, ImageItem> uniquePages = new TreeMap<>();
                for (ImageItem item : selected) uniquePages.put(item.suffixNumber, item);
                selected = new ArrayList<>(uniquePages.values());
                Collections.sort(selected, Comparator.comparingInt(a -> a.suffixNumber));

                String albumName = "PPT_Notein_Strict_" +
                        new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());

                int total = selected.size();
                int ok = 0;

                Calendar anchor = Calendar.getInstance();
                anchor.set(Calendar.HOUR_OF_DAY, 20);
                anchor.set(Calendar.MINUTE, 0);
                anchor.set(Calendar.SECOND, 0);
                anchor.set(Calendar.MILLISECOND, 0);
                long baseTime = anchor.getTimeInMillis();

                // 仍然倒序插入，让第一页获得最大的 MediaStore _ID；
                // 但所有页的逻辑时间固定在同一天 20:00 附近，每页只差 1 秒。
                for (int i = total - 1; i >= 0; i--) {
                    ImageItem item = selected.get(i);
                    long pageTime = baseTime - ((long) i * 1000L);
                    if (copyToGallery(item, albumName, pageTime)) ok++;

                    final int done = total - i;
                    final String name = item.name;
                    runOnUiThread(() -> statusText.setText(
                            "正在生成同日严格排序 " + done + "/" + total + "\n" + name));
                }

                final int success = ok;
                runOnUiThread(() -> {
                    statusText.setText(
                            "完成！成功 " + success + "/" + total + " 张。\n\n" +
                            "新相册：Pictures/" + albumName + "\n" +
                            "本版已强制：\n" +
                            "• 所有图片位于同一天\n" +
                            "• 每页仅相差 1 秒\n" +
                            "• 001/002/003 文件名 + TITLE\n" +
                            "• DATE_TAKEN / DATE_ADDED / DATE_MODIFIED\n" +
                            "• EXIF DateTimeOriginal / DateTimeDigitized\n" +
                            "• 第一页最后写入，获得最大 MediaStore _ID\n\n" +
                            "请完全退出 Notein 后重新打开，再进入图片选择器查看这个新相册。");
                    startButton.setEnabled(true);
                    Toast.makeText(this, "v1.3 相册已生成", Toast.LENGTH_LONG).show();
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    statusText.setText("处理失败：\n" + e.getClass().getSimpleName() + ": " + e.getMessage());
                    startButton.setEnabled(true);
                });
            }
        }).start();
    }

    private List<ImageItem> scanFolder(Uri treeUri, boolean zeroBased) throws Exception {
        List<ImageItem> result = new ArrayList<>();
        String treeDocId = DocumentsContract.getTreeDocumentId(treeUri);
        Uri childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, treeDocId);
        String[] projection = new String[]{
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_MIME_TYPE
        };

        try (Cursor cursor = getContentResolver().query(childrenUri, projection, null, null, null)) {
            if (cursor == null) return result;
            int idIndex = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DOCUMENT_ID);
            int nameIndex = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DISPLAY_NAME);
            int mimeIndex = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_MIME_TYPE);

            while (cursor.moveToNext()) {
                String docId = cursor.getString(idIndex);
                String name = cursor.getString(nameIndex);
                String mime = cursor.getString(mimeIndex);
                if (name == null) continue;

                Matcher m = PAGE_SUFFIX.matcher(name);
                if (!m.find()) continue;

                int suffix = Integer.parseInt(m.group(1));
                int page = zeroBased ? suffix + 1 : suffix;
                if (page <= 0) continue;

                ImageItem item = new ImageItem();
                item.uri = DocumentsContract.buildDocumentUriUsingTree(treeUri, docId);
                item.name = name;
                item.mime = (mime != null && mime.startsWith("image/")) ? mime : guessMime(m.group(2));
                item.ext = normalizeExt(m.group(2));
                item.suffixNumber = suffix;
                item.pageNumber = page;
                item.batchKey = name.substring(0, m.start());
                item.batchId = extractBatchId(item.batchKey);
                result.add(item);
            }
        }
        return result;
    }

    private boolean copyToGallery(ImageItem item, String albumName, long pageTime) {
        ContentResolver resolver = getContentResolver();
        Uri outputUri = null;
        try {
            String pageLabel = String.format(Locale.US, "%03d", item.pageNumber);
            String displayName = pageLabel + "." + item.ext;

            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.DISPLAY_NAME, displayName);
            values.put(MediaStore.Images.Media.TITLE, pageLabel);
            values.put(MediaStore.Images.Media.DESCRIPTION, "PPT page " + pageLabel);
            values.put(MediaStore.Images.Media.MIME_TYPE, item.mime);
            values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/" + albumName);
            values.put(MediaStore.Images.Media.DATE_TAKEN, pageTime);
            values.put(MediaStore.Images.Media.DATE_ADDED, pageTime / 1000L);
            values.put(MediaStore.Images.Media.DATE_MODIFIED, pageTime / 1000L);
            values.put(MediaStore.Images.Media.IS_PENDING, 1);

            outputUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            if (outputUri == null) return false;

            try (InputStream in = resolver.openInputStream(item.uri);
                 OutputStream out = resolver.openOutputStream(outputUri, "w")) {
                if (in == null || out == null) throw new Exception("无法打开图片流");
                byte[] buffer = new byte[128 * 1024];
                int n;
                while ((n = in.read(buffer)) > 0) out.write(buffer, 0, n);
                out.flush();
            }

            if ("jpg".equalsIgnoreCase(item.ext) || "jpeg".equalsIgnoreCase(item.ext)) {
                try (ParcelFileDescriptor pfd = resolver.openFileDescriptor(outputUri, "rw")) {
                    if (pfd != null) {
                        ExifInterface exif = new ExifInterface(pfd.getFileDescriptor());
                        SimpleDateFormat exifFmt = new SimpleDateFormat("yyyy:MM:dd HH:mm:ss", Locale.US);
                        exifFmt.setTimeZone(TimeZone.getDefault());
                        String t = exifFmt.format(new Date(pageTime));
                        exif.setAttribute(ExifInterface.TAG_DATETIME, t);
                        exif.setAttribute(ExifInterface.TAG_DATETIME_ORIGINAL, t);
                        exif.setAttribute(ExifInterface.TAG_DATETIME_DIGITIZED, t);
                        exif.setAttribute(ExifInterface.TAG_IMAGE_DESCRIPTION, "PPT page " + pageLabel);
                        exif.saveAttributes();
                    }
                } catch (Exception ignored) {}
            }

            ContentValues done = new ContentValues();
            done.put(MediaStore.Images.Media.IS_PENDING, 0);
            done.put(MediaStore.Images.Media.TITLE, pageLabel);
            done.put(MediaStore.Images.Media.DATE_TAKEN, pageTime);
            done.put(MediaStore.Images.Media.DATE_ADDED, pageTime / 1000L);
            done.put(MediaStore.Images.Media.DATE_MODIFIED, pageTime / 1000L);
            resolver.update(outputUri, done, null, null);
            return true;
        } catch (Exception e) {
            if (outputUri != null) {
                try { resolver.delete(outputUri, null, null); } catch (Exception ignored) {}
            }
            return false;
        }
    }

    private long extractBatchId(String batchKey) {
        Matcher matcher = Pattern.compile("(\\d{10,})").matcher(batchKey);
        long value = 0L;
        while (matcher.find()) {
            try { value = Long.parseLong(matcher.group(1)); }
            catch (NumberFormatException ignored) {}
        }
        return value;
    }

    private String normalizeExt(String ext) {
        String e = ext.toLowerCase(Locale.US);
        return "jpeg".equals(e) ? "jpg" : e;
    }

    private String guessMime(String ext) {
        String e = ext.toLowerCase(Locale.US);
        if ("png".equals(e)) return "image/png";
        if ("webp".equals(e)) return "image/webp";
        return "image/jpeg";
    }
}
