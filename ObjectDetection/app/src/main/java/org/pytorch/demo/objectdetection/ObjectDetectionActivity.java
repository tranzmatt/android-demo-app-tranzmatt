package org.pytorch.demo.objectdetection;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.ViewStub;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ObjectDetectionActivity extends AbstractCameraXActivity<ObjectDetectionActivity.AnalysisResult> {
    private Module mModule = null;
    private ResultView mResultView;
    private TextView mTextView;

    static class AnalysisResult {
        private final ArrayList<Result> mResults;
        private final String mTextResult;

        public AnalysisResult(ArrayList<Result> results, String textResult) {
            mResults = results;
            mTextResult = textResult;
        }
    }

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_object_detection;
    }

    @Override
    protected PreviewView getCameraPreviewTextureView() {
        mResultView = findViewById(R.id.resultView);
        mTextView = findViewById(R.id.textResultView);
        return (PreviewView) ((ViewStub) findViewById(R.id.object_detection_texture_view_stub))
                .inflate()
                .findViewById(R.id.object_detection_texture_view);
    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        Log.d("Object Detection", "Applying results to UI: " + result.mTextResult);
        mResultView.setResults(result.mResults);
        mResultView.invalidate();
        mTextView.setText(result.mTextResult);
        mTextView.invalidate();
    }

    private Bitmap imgToBitmap(Image image) {
        if (image == null) {
            Log.e("Object Detection", "Image is null");
            return null;
        }

        Image.Plane[] planes = image.getPlanes();
        if (planes.length != 3) {
            Log.e("Object Detection", "Image does not have 3 planes");
            return null;
        }

        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        boolean success = yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        if (!success) {
            Log.e("Object Detection", "Failed to compress YUV image to JPEG");
            return null;
        }

        byte[] imageBytes = out.toByteArray();
        Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);

        if (bitmap == null) {
            Log.e("Object Detection", "Failed to decode byte array to Bitmap");
        } else {
            Log.d("Object Detection", "Bitmap successfully decoded from byte array");
        }

        return bitmap;
    }


    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        try {
            if (mModule == null) {
                mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "yolov5s.torchscript.ptl"));
                Log.d("Object Detection", "Model loaded successfully");
            }

            // Check if mClasses is already initialized
            if (PrePostProcessor.mClasses == null || PrePostProcessor.mClasses.length == 0) {
                // Load class names from classes.txt
                BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("classes.txt")));
                String line;
                List<String> classes = new ArrayList<>();
                while ((line = br.readLine()) != null) {
                    classes.add(line);
                }
                PrePostProcessor.mClasses = new String[classes.size()];
                classes.toArray(PrePostProcessor.mClasses);
                br.close();
                Log.d("Object Detection", "Classes loaded successfully");

                // Print the loaded classes
                Log.d("Object Detection", "Loaded classes: " + Arrays.toString(PrePostProcessor.mClasses));
            }
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            return null;
        }

        Image mediaImage = image.getImage();
        if (mediaImage == null) {
            Log.e("Object Detection", "Failed to get media image from ImageProxy");
            return null;
        }

        Bitmap bitmap = imgToBitmap(mediaImage);
        if (bitmap == null) {
            Log.e("Object Detection", "Failed to convert image to bitmap");
            return null;
        }

        Log.d("Object Detection", "Bitmap created from Image: " + bitmap.getWidth() + "x" + bitmap.getHeight());
        saveBitmap(bitmap, "original_bitmap.jpg");

        Matrix matrix = new Matrix();
        matrix.postRotate(rotationDegrees);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        Log.d("Object Detection", "Bitmap rotated");
        saveBitmap(bitmap, "rotated_bitmap.jpg");

        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);
        Log.d("Object Detection", "Bitmap resized: " + resizedBitmap.getWidth() + "x" + resizedBitmap.getHeight());
        saveBitmap(resizedBitmap, "resized_bitmap.jpg");

        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
        Log.d("Object Detection", "Input tensor created");

        IValue[] outputTuple = mModule.forward(IValue.from(inputTensor)).toTuple();
        final Tensor outputTensor = outputTuple[0].toTensor();
        final float[] outputs = outputTensor.getDataAsFloatArray();
        Log.d("Object Detection", "Model inference completed");

        float imgScaleX = (float) bitmap.getWidth() / PrePostProcessor.mInputWidth;
        float imgScaleY = (float) bitmap.getHeight() / PrePostProcessor.mInputHeight;
        float ivScaleX = (float) mResultView.getWidth() / bitmap.getWidth();
        float ivScaleY = (float) mResultView.getHeight() / bitmap.getHeight();

        final ArrayList<Result> results = PrePostProcessor.outputsToNMSPredictions(outputs, imgScaleX, imgScaleY, ivScaleX, ivScaleY, 0, 0);
        Log.d("Object Detection", "NMS Predictions computed: " + results.size() + " results");

        StringBuilder resultText = new StringBuilder();
        for (Result res : results) {
            resultText.append(String.format("%s: %.2f\n", PrePostProcessor.mClasses[res.classIndex], res.score));
        }

        Log.d("Object Detection", "Detection results: " + resultText.toString());

        return new AnalysisResult(results, resultText.toString());
    }

    private void saveBitmap(Bitmap bitmap, String fileName) {
        ContentResolver resolver = getContentResolver();
        ContentValues contentValues = new ContentValues();
        contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
        contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg");
        contentValues.put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/ObjectDetection");

        OutputStream fos = null;
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                fos = resolver.openOutputStream(resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues));
            } else {
                String imagesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).toString() + "/ObjectDetection";
                File file = new File(imagesDir);
                if (!file.exists()) {
                    file.mkdir();
                }
                File image = new File(imagesDir, fileName);
                fos = new FileOutputStream(image);
            }
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            Log.d("Object Detection", "Saved bitmap to " + fileName);
        } catch (IOException e) {
            Log.e("Object Detection", "Failed to save bitmap", e);
        } finally {
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    Log.e("Object Detection", "Failed to close output stream", e);
                }
            }
        }
    }
}
