package com.example.yolov5tfliteandroid;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Point;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.camera.lifecycle.ProcessCameraProvider;

import com.example.yolov5tfliteandroid.analysis.FullScreenAnalyse;
import com.example.yolov5tfliteandroid.detector.RobustVideoMattingTFLiteDetector;
import com.example.yolov5tfliteandroid.utils.CameraProcess;
import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;

import com.example.yolov5tfliteandroid.utils.BackgroundResize;


public class MainActivity extends AppCompatActivity {

    private boolean IS_FULL_SCREEN = false;

    private PreviewView cameraPreviewMatch;
    private ImageView boxLabelCanvas;
    private TextView inferenceTimeTextView;
    private TextView frameSizeTextView;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private RobustVideoMattingTFLiteDetector robustVideoMattingTFLiteDetector;

    private CameraProcess cameraProcess = new CameraProcess();

    /**
     * 获取屏幕旋转角度,0表示拍照出来的图片是横屏
     *
     */
    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    /**
     * 加载模型
     *
     * @param modelName
     */
    private void initModel(String modelName) {
        // 加载模型
        try {
            this.robustVideoMattingTFLiteDetector = new RobustVideoMattingTFLiteDetector();
            this.robustVideoMattingTFLiteDetector.setModelFile(modelName);
//            this.yolov5TFLiteDetector.addNNApiDelegate();
            //this.robustVideoMattingTFLiteDetector.addGPUDelegate();
            //0627修改 这里直接改成线程加速
            this.robustVideoMattingTFLiteDetector.threadFast();
            this.robustVideoMattingTFLiteDetector.initialModel(this);
            Log.i("model", "Success loading model" + this.robustVideoMattingTFLiteDetector.getModelFile());
        } catch (Exception e) {
            Log.e("image", "load model error: " + e.getMessage() + e.toString());
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        //启动openCV
        OpenCVLoader.initDebug();
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 打开app的时候隐藏顶部状态栏
//        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE | View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN);
        getWindow().setStatusBarColor(Color.TRANSPARENT);

        // 全屏画面
        cameraPreviewMatch = findViewById(R.id.camera_preview_match);
        // 下面这行可以让原始的全屏画面不显示
        cameraPreviewMatch.setVisibility(View.INVISIBLE);
        cameraPreviewMatch.setScaleType(PreviewView.ScaleType.FILL_START);

//        //**************************************0717修改************************************************************
//        DisplayMetrics displayMetrics = new DisplayMetrics();
//        WindowManager windowManager = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
//        Display display = windowManager.getDefaultDisplay();
//        Point screenSize = new Point();
//        display.getSize(screenSize);
//        int screenWidth = screenSize.x;
//        int screenHeight = screenSize.y;
//
//        //**************************************************************************************************


        // box/label画面
        boxLabelCanvas = findViewById(R.id.box_label_canvas);

//      **********************7.16日添加 设置imageview控件长宽比16:9全屏显示****************************
        ViewGroup.LayoutParams layoutParams = boxLabelCanvas.getLayoutParams();
        int desiredWidth = getResources().getDisplayMetrics().widthPixels;

        int desiredHeight = getResources().getDisplayMetrics().heightPixels;
        // 计算调整后的宽度和高度
        int adjustedWidth, adjustedHeight;
        adjustedWidth = desiredWidth;
        adjustedHeight = (int) (desiredHeight / 0.5625);
        layoutParams.width=adjustedHeight;
        layoutParams.height=adjustedWidth;
        boxLabelCanvas.setLayoutParams(layoutParams);
//***********************************************************************************************

        // 实时更新的一些view
        inferenceTimeTextView = findViewById(R.id.inference_time);
        frameSizeTextView = findViewById(R.id.frame_size);
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        // 申请摄像头权限
        if (!cameraProcess.allPermissionsGranted(this)) {
            cameraProcess.requestPermissions(this);
        }

        // 获取手机摄像头拍照旋转参数
        int rotation = getWindowManager().getDefaultDisplay().getRotation();
        Log.i("image", "rotation: " + rotation);

        cameraProcess.showCameraSupportSize(MainActivity.this);

        // 初始化加载RobustVideoMatting
        initModel("RobustVideoMatting");

//        int previewHeight = cameraPreviewMatch.getHeight();
//        int previewWidth = cameraPreviewMatch.getWidth();
        int previewHeight = 360;  //16:9的时候设置360  4:3的时候设置480
        int previewWidth = 640;

        // 背景图片  你可以自己换 打开res.drawable 里面有可供选择的图片
        Bitmap origin_background = BitmapFactory.decodeResource(getResources(),R.drawable.huiyishi1);
        // 因为背景图片的大小不确定 为了后续速度的提升 这里先把图像进行缩放 再送入到后面画面进行处理
        BackgroundResize resizefuc=new BackgroundResize();
        Bitmap background = resizefuc.zoomImage(origin_background,previewWidth,previewHeight);


        FullScreenAnalyse fullScreenAnalyse = new FullScreenAnalyse(MainActivity.this,
                cameraPreviewMatch,
                boxLabelCanvas,
                rotation,
                inferenceTimeTextView,
                frameSizeTextView,
                robustVideoMattingTFLiteDetector,
                background);
        cameraProcess.startCamera(MainActivity.this, fullScreenAnalyse, cameraPreviewMatch);


    }
}