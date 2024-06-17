package com.example.yolov5tfliteandroid.detector;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;


public class RobustVideoMattingTFLiteDetector {

    //改前private final Size INPNUT_SIZE = new Size(448, 224);
    private final Size INPNUT_SIZE = new Size(160, 90);

   //改前 private final int[] OUTPUT_SIZE = new int[]{1, 224, 448,1};
    private final int[] OUTPUT_SIZE = new int[]{1, 90, 160,1};
    private Boolean IS_INT8 = false;
    private final String MODEL_fenge =  "0623tflite.tflite";
    private String MODEL_FILE;

    private Interpreter tflite;
    //private List<String> associatedAxisLabels;
    Interpreter.Options options = new Interpreter.Options();


    public String getModelFile() {
        return this.MODEL_FILE;
    }

    public void setModelFile(String modelFile){
        IS_INT8 = false;
        MODEL_FILE = MODEL_fenge;
    }


    public Size getInputSize(){return this.INPNUT_SIZE;}
    public int[] getOutputSize(){return this.OUTPUT_SIZE;}

    /**
     * 初始化模型, 可以通过 addNNApiDelegate(), addGPUDelegate()提前加载相应代理
     *
     * @param activity
     */
    public void initialModel(Context activity) {
        // Initialise the model
        try {

            ByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, MODEL_FILE);
            tflite = new Interpreter(tfliteModel, options);
            Log.i("tfliteSupport", "Success reading model: " + MODEL_FILE);



        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading model or label: ", e);
            Toast.makeText(activity, "load model error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    /**
     * 检测步骤
     *
     * @param bitmap
     * @return
     */
    public float[] detect(Bitmap bitmap) {

        // yolov5s-tflite的输入是:[1, 320, 320,3], 摄像头每一帧图片需要resize,再归一化

        // 新模型RobustVideoMatting的输入是1 90 160 3
        TensorImage RobustVideoMattingTfliteInput;    //input容器
        ImageProcessor imageProcessor;

        //Float32 所用
        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(0, 255))
                .build();
        RobustVideoMattingTfliteInput = new TensorImage(DataType.FLOAT32);

//        //INT8 所用
//        imageProcessor = new ImageProcessor.Builder()
//                .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
//                .add(new NormalizeOp(0, 255))
//                .build();
//        RobustVideoMattingTfliteInput = new TensorImage(DataType.UINT8);


        RobustVideoMattingTfliteInput.load(bitmap);
        RobustVideoMattingTfliteInput = imageProcessor.process(RobustVideoMattingTfliteInput);


        // 输出存放处
        TensorBuffer probabilityBuffer;
        //Float32 所用
        probabilityBuffer = TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.FLOAT32);
        //INT8 所用
        //probabilityBuffer = TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.UINT8);

        // 推理计算
        if (null != tflite) {
            // 这里tflite默认会加一个batch=1的纬度
            tflite.run(RobustVideoMattingTfliteInput.getBuffer(), probabilityBuffer.getBuffer());
        }

        // 输出数据被平铺了出来  [ 1 , 224, 448 , 1]
        float[] RobustVideoMattingresult = probabilityBuffer.getFloatArray();

        return RobustVideoMattingresult;
    }



    /**
     * 添加NNapi代理
     */
    public void addNNApiDelegate() {
        NnApiDelegate nnApiDelegate = null;
        // Initialize interpreter with NNAPI delegate for Android Pie or above
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
//            NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
//            nnApiOptions.setAllowFp16(true);
//            nnApiOptions.setUseNnapiCpu(true);
            //ANEURALNETWORKS_PREFER_LOW_POWER：倾向于以最大限度减少电池消耗的方式执行。这种设置适合经常执行的编译。
            //ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER：倾向于尽快返回单个答案，即使这会耗费更多电量。这是默认值。
            //ANEURALNETWORKS_PREFER_SUSTAINED_SPEED：倾向于最大限度地提高连续帧的吞吐量，例如，在处理来自相机的连续帧时。
//            nnApiOptions.setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED);
//            nnApiDelegate = new NnApiDelegate(nnApiOptions);
            nnApiDelegate = new NnApiDelegate();
            options.addDelegate(nnApiDelegate);
            Log.i("tfliteSupport", "using nnapi delegate.");
        }
    }

    /**
     * 添加GPU代理
     */
    public void addGPUDelegate() {
        CompatibilityList compatibilityList = new CompatibilityList();
        if(compatibilityList.isDelegateSupportedOnThisDevice()){
            GpuDelegate.Options delegateOptions = compatibilityList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            options.addDelegate(gpuDelegate);
            Log.i("zhangmingjianGPU", "使用GPU成功");
            Log.i("tfliteSupport", "using gpu delegate.");
        } else {
            addThread(4);
            Log.i("zhangmingjianGPU", "使用GPU失败");
        }

        //直接用线程加速
        //addThread(4);
    }

    /**
     * 直接线程加速
     */
    public void threadFast(){
        addThread(4);
    }

    /**
     * 添加线程数
     * @param thread
     */
    public void addThread(int thread) {
        options.setNumThreads(thread);
    }

}
