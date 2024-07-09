package org.pytorch.demo.objectdetection;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
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
        Image.Plane[] planes = image.getPlanes();
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
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
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
            // Load class names from classes.txt
            // Check if mClasses is already initialized
            if (PrePostProcessor.mClasses == null || PrePostProcessor.mClasses.length == 0) {
                // Load class names from classes.txt
                BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("classes.txt")));
                String line;
                List<String> classes = new ArrayList<>();
                while ((line = br.readLine()) != null) {
                    classes.add(line);
                    Log.d("Class Load", "Loaded class " + line);
                }
                PrePostProcessor.mClasses = new String[classes.size()];
                classes.toArray(PrePostProcessor.mClasses);
                br.close();
            }
            // Print the loaded classes
            Log.d("Object Detection", "Loaded classes: " + Arrays.toString(PrePostProcessor.mClasses));

        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            return null;
        }

        Bitmap bitmap = imgToBitmap(image.getImage());
        if (bitmap == null) {
            Log.e("Object Detection", "Failed to convert image to bitmap");
            return null;
        }

        Log.d("Object Detection", "Bitmap created from Image: " + bitmap.getWidth() + "x" + bitmap.getHeight());

        Matrix matrix = new Matrix();
        matrix.postRotate(rotationDegrees);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        Log.d("Object Detection", "Bitmap rotated");

        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);
        Log.d("Object Detection", "Bitmap resized: " + resizedBitmap.getWidth() + "x" + resizedBitmap.getHeight());

        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
        Log.d("Object Detection", "Input tensor created");

        // Log the input tensor values
        float[] inputTensorData = inputTensor.getDataAsFloatArray();
        Log.d("Object Detection", "Input tensor values: " + Arrays.toString(inputTensorData));

        IValue[] outputTuple = mModule.forward(IValue.from(inputTensor)).toTuple();
        final Tensor outputTensor = outputTuple[0].toTensor();
        final float[] outputs = outputTensor.getDataAsFloatArray();
        Log.d("Object Detection", "Model inference completed");

        float imgScaleX = (float)bitmap.getWidth() / PrePostProcessor.mInputWidth;
        float imgScaleY = (float)bitmap.getHeight() / PrePostProcessor.mInputHeight;
        float ivScaleX = (float)mResultView.getWidth() / bitmap.getWidth();
        float ivScaleY = (float)mResultView.getHeight() / bitmap.getHeight();

        final ArrayList<Result> results = PrePostProcessor.outputsToNMSPredictions(outputs, imgScaleX, imgScaleY, ivScaleX, ivScaleY, 0, 0);
        Log.d("Object Detection", "NMS Predictions computed: " + results.size() + " results");

        StringBuilder resultText = new StringBuilder();
        for (Result res : results) {
            resultText.append(String.format("%s: %.2f\n", PrePostProcessor.mClasses[res.classIndex], res.score));
        }

        Log.d("Object Detection", "Detection results: " + resultText.toString());

        return new AnalysisResult(results, resultText.toString());
    }
}
