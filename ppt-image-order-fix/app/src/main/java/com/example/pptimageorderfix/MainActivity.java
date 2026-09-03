package com.example.pptimageorderfix;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;
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
        title.setText("PPT 图片顺序修复");
        title.setTextSize(26);
        title.setPadding(0, 0, 0, dp(12));
        root.addView(title);

        TextView desc = new TextView(this);
        desc.setText(
                "专门处理 QQ 浏览器 PPT 转图片后的顺序问题。\n\n" +
                "它会：\n" +
                "1. 读取文件名最后的 _0、_1、_2…\n" +
                "2. 按数字而不是文字排序\n" +
                "3. 复制成 001.jpg、002.jpg、003.jpg…\n" +
                "4. 写入新的系统相册，并设置时间顺序，让小米相册更稳定地按 PPT 页码显示\n\n" +
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

        TextView hint = new TextView(this);
        hint.setText("你截图里的情况应保持勾选。如果以后遇到 _1 就是第1页的文件，再取消勾选。");
        hint.setTextSize(13);
        hint.setPadding(0, 0, 0, dp(10));
        root.addView(hint);

        startButton = new Button(this);
        startButton.setText("② 一键整理到新相册");
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
        float density = getResources().getDisplayMetrics().density;
        return Math.round(value * density);
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

            int flags = data.getFlags() &
                    (Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            try {
                getContentResolver().takePersistableUriPermission(
                        uri, flags & Intent.FLAG_GRANT_READ_URI_PERMISSION);
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

                if (items.isEmpty()) {
                    runOnUiThread(() -> {
                        statusText.setText("没有找到文件名以 _数字.jpg / png / webp 结尾的图片。");
                        startButton.setEnabled(true);
                    });
                    return;
                }

                Collections.sort(items, Comparator.comparingInt(a -> a.suffixNumber));

                String albumName = "PPT_Ordered_" +
                        new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());

                long baseTime = System.currentTimeMillis();
                int total = items.size();
                int ok = 0;

                for (int i = 0; i < total; i++) {
                    ImageItem item = items.get(i);
                    long takenTime = baseTime - i * 1000L;
                    if (copyToGallery(item, albumName, takenTime)) {
                        ok++;
                    }

                    final int done = i + 1;
                    final String name = item.name;
                    runOnUiThread(() ->
                            statusText.setText("正在整理 " + done + "/" + total + "\n" + name));
                }

                final int success = ok;
                runOnUiThread(() -> {
                    statusText.setText(
                            "完成！成功 " + success + "/" + total + " 张。\n\n" +
                            "新相册位置：Pictures/" + albumName + "\n" +
                            "文件名已整理为 001、002、003…\n" +
                            "原文件未修改。");
                    startButton.setEnabled(true);
                    Toast.makeText(this, "整理完成", Toast.LENGTH_LONG).show();
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

        String[] projection = new String[] {
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
                result.add(item);
            }
        }

        return result;
    }

    private boolean copyToGallery(ImageItem item, String albumName, long takenTime) {
        ContentResolver resolver = getContentResolver();
        Uri outputUri = null;

        try {
            String displayName = String.format(Locale.US, "%03d.%s", item.pageNumber, item.ext);

            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.DISPLAY_NAME, displayName);
            values.put(MediaStore.Images.Media.MIME_TYPE, item.mime);
            values.put(MediaStore.Images.Media.RELATIVE_PATH,
                    Environment.DIRECTORY_PICTURES + "/" + albumName);
            values.put(MediaStore.Images.Media.DATE_TAKEN, takenTime);
            values.put(MediaStore.Images.Media.DATE_ADDED, takenTime / 1000L);
            values.put(MediaStore.Images.Media.DATE_MODIFIED, takenTime / 1000L);
            values.put(MediaStore.Images.Media.IS_PENDING, 1);

            outputUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            if (outputUri == null) return false;

            try (InputStream in = resolver.openInputStream(item.uri);
                 OutputStream out = resolver.openOutputStream(outputUri, "w")) {
                if (in == null || out == null) throw new Exception("无法打开图片流");
                byte[] buffer = new byte[128 * 1024];
                int n;
                while ((n = in.read(buffer)) > 0) {
                    out.write(buffer, 0, n);
                }
                out.flush();
            }

            ContentValues done = new ContentValues();
            done.put(MediaStore.Images.Media.IS_PENDING, 0);
            done.put(MediaStore.Images.Media.DATE_TAKEN, takenTime);
            done.put(MediaStore.Images.Media.DATE_MODIFIED, takenTime / 1000L);
            resolver.update(outputUri, done, null, null);
            return true;

        } catch (Exception e) {
            if (outputUri != null) {
                try { resolver.delete(outputUri, null, null); } catch (Exception ignored) {}
            }
            return false;
        }
    }

    private String normalizeExt(String ext) {
        String e = ext.toLowerCase(Locale.US);
        if ("jpeg".equals(e)) return "jpg";
        return e;
    }

    private String guessMime(String ext) {
        String e = ext.toLowerCase(Locale.US);
        if ("png".equals(e)) return "image/png";
        if ("webp".equals(e)) return "image/webp";
        return "image/jpeg";
    }
}
