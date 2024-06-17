package com.example.yolov5tfliteandroid.analysis;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_32FC3;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgproc.Imgproc.COLOR_BGRA2BGR;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;

import com.example.yolov5tfliteandroid.detector.RobustVideoMattingTFLiteDetector;
import com.example.yolov5tfliteandroid.utils.ImageProcess;

import io.reactivex.rxjava3.android.schedulers.AndroidSchedulers;
import io.reactivex.rxjava3.core.Observable;
import io.reactivex.rxjava3.core.ObservableEmitter;
import io.reactivex.rxjava3.schedulers.Schedulers;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;


public class FullScreenAnalyse implements ImageAnalysis.Analyzer {

    public static class Result{

        public Result(long costTime, Bitmap bitmap, int outHeight,int outWidth) {
            this.costTime = costTime;
            this.bitmap = bitmap;
            this.outHeight = outHeight;
            this.outWidth = outWidth;
        }
        long costTime;
        Bitmap bitmap;
        int outHeight;
        int outWidth;
    }

    ImageView boxLabelCanvas;
    PreviewView previewView;
    int rotation;
    private TextView inferenceTimeTextView;
    private TextView frameSizeTextView;
    ImageProcess imageProcess;
    private RobustVideoMattingTFLiteDetector robustVideoMattingTFLiteDetector;
    private Bitmap background;

    public FullScreenAnalyse(Context context,
                             PreviewView previewView,
                             ImageView boxLabelCanvas,
                             int rotation,
                             TextView inferenceTimeTextView,
                             TextView frameSizeTextView,
                             RobustVideoMattingTFLiteDetector robustVideoMattingTFLiteDetector,
                             Bitmap background) {
        this.previewView = previewView;
        this.boxLabelCanvas = boxLabelCanvas;
        this.rotation = rotation;
        this.inferenceTimeTextView = inferenceTimeTextView;
        this.frameSizeTextView = frameSizeTextView;
        this.imageProcess = new ImageProcess();
        this.robustVideoMattingTFLiteDetector = robustVideoMattingTFLiteDetector;
        this.background=background;
    }


    @Override
    public void analyze(@NonNull ImageProxy image) {

        // 这里Observable将image analyse的逻辑放到子线程计算, 渲染UI的时候再拿回来对应的数据, 避免前端UI卡顿
        Observable.create( (ObservableEmitter<Result> emitter) -> {
            long start = System.currentTimeMillis();

            byte[][] yuvBytes = new byte[3][];
            ImageProxy.PlaneProxy[] planes = image.getPlanes();
            int imageHeight = image.getHeight();
            int imagewWidth = image.getWidth();

            imageProcess.fillBytes(planes, yuvBytes);
            int yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            int[] rgbBytes = new int[imageHeight * imagewWidth];
            imageProcess.YUV420ToARGB8888(
                    yuvBytes[0],
                    yuvBytes[1],
                    yuvBytes[2],
                    imagewWidth,
                    imageHeight,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbBytes);


            // 原图bitmap
            Bitmap imageBitmap = Bitmap.createBitmap(imagewWidth, imageHeight, Bitmap.Config.ARGB_8888);

            imageBitmap.setPixels(rgbBytes, 0, imagewWidth, 0, 0, imagewWidth, imageHeight);




            // 因为旋转了 之前的高是360 宽是640
            // 但是后面要变成 竖屏幕 即高是640 宽是360
            // 这里设置outheight和outwidth
            int outHeight=360;   //16:9的时候设置360  4:3的时候设置480
            int outWidth=640;    //640

           // -----------------------------------------------------------------------
            // 模型输入的bitmap
            Matrix previewToModelTransform =
                    imageProcess.getTransformationMatrix(
                            imageBitmap.getWidth(), imageBitmap.getHeight(),
                            robustVideoMattingTFLiteDetector.getInputSize().getWidth(),
                            robustVideoMattingTFLiteDetector.getInputSize().getHeight(),
                            0, false);


            Bitmap modelInputBitmap = Bitmap.createBitmap(imageBitmap, 0, 0,
                    imageBitmap.getWidth(), imageBitmap.getHeight(),
                    previewToModelTransform, false);
            //90*160
            // -----------------------------------------------------------------------

//            // 模型输入的bitmap
//            Matrix previewToModelTransform =
//                    imageProcess.getTransformationMatrix(
//                            cropImageBitmap.getWidth(), cropImageBitmap.getHeight(),
//                            robustVideoMattingTFLiteDetector.getInputSize().getWidth(),
//                            robustVideoMattingTFLiteDetector.getInputSize().getHeight(),
//                            0, false);
//
//
//            Bitmap modelInputBitmap = Bitmap.createBitmap(cropImageBitmap, 0, 0,
//                    cropImageBitmap.getWidth(), cropImageBitmap.getHeight(),
//                    previewToModelTransform, false);
//            //90*160


//            //使用前置：镜像翻转
//            Mat orin_input1 = new Mat();
//            Utils.bitmapToMat(modelInputBitmap, orin_input1);
//            Core.flip(orin_input1, orin_input1,0);
//            Utils.matToBitmap(orin_input1,modelInputBitmap);
//            //使用前置：镜像翻转
//            Mat orin_input2 = new Mat();
//            Utils.bitmapToMat(cropImageBitmap, orin_input2);
//            Core.flip(orin_input2, orin_input2,0);
//            Utils.matToBitmap(orin_input2,cropImageBitmap);


            long begin = System.currentTimeMillis();
            long qianchuli = begin - start;


            Log.i("wangqian", "before_time：" + qianchuli);
            // 丢到模型里面去做检测
            float[] result = robustVideoMattingTFLiteDetector.detect(modelInputBitmap);
            //输出一维数组90*160 就是segmap结果

//            //搞的输出值测试
//            float[] test= new float[result.length/2];
//            int j=0;
//            for(int i=result.length/2;i<result.length;i++){
//                test[j]=result[i];
//            }
//            int a=1;


            long last = System.currentTimeMillis();
            long time = (last - begin);

            Log.i("wangqian", "模型推理时延：" + time);
            //后处理
            long houchuli_start = System.currentTimeMillis();




//
//            //<---------------------6.14添加-------------------->
            //这里是crop后的屏幕大小  previewWidth, previewHeight
            //imagewWidth, imageHeight

            Mat fg = new Mat(outHeight,outWidth,CV_8UC3);

            Mat bg = new Mat(outHeight,outWidth,CV_8UC3);
            //这里的background是背景图片 backgroundBitmap是裁剪后的图片 进行后续操作


            Mat mskmat = new Mat (90, 160, CV_32F);
            Mat invmskmat = new Mat(outHeight, outWidth,CV_32FC3, new Scalar(1.0,1.0,1.0));

            mskmat.put(0,0,result);

//            //-----------------------------------------------------------------------------
            //0626尝试修改------------------------------------------------------------------
            Core.subtract(mskmat,new Scalar(0.65),mskmat);
            Imgproc.threshold(mskmat,mskmat,0,1,Imgproc.THRESH_BINARY); //就是gray_image_data

//            //后面其实没啥用
//            Mat mask = new Mat();
//            Core.compare(mskmat,new Scalar(1),mask,Core.CMP_EQ); //获得了mask
//            Mat color_image = new Mat (90, 160, CV_32F); //color_image全是0
//            Mat cmap0 = new Mat (90, 160, CV_32F); //cmap0全是255
//            Mat a = new Mat();
//
//            //这里就是改写 a=np.where(mask,cmap0,color_image)
//            Core.bitwise_and(cmap0, mask, a); // 对颜色映射矩阵和掩码矩阵进行按位与操作，将结果保存到a矩阵中
//            Core.bitwise_not(mask, mask); // 将掩码矩阵取反
//            Core.bitwise_and(color_image, mask, mask); // 对彩色图像矩阵和取反后的掩码矩阵进行按位与操作，将结果保存到mask矩阵中
//            Core.bitwise_or(a, mask, a); // 对a矩阵和mask矩阵进行按位或操作，将结果保存到a矩阵中
//            mskmat = a;
//
//
//            //-----------------------------------------------------------------------------
//            //-----------------------------------------------------------------------------


            //下面是原来的 建议解放Imgproc.threshold(mskmat, mskmat,1.0,1.0,Imgproc.THRESH_TRUNC);
            //Core.add(mskmat,new Scalar(0.65),mskmat);     //0516 112112需要+0.5     0613 90160不需要+0.5
            //Core.multiply(mskmat,new Scalar(2.0),mskmat);
            //將mskmat>1全部变成1，<1的全部保持不动
            //Imgproc.threshold(mskmat, mskmat,1.0,1.0,Imgproc.THRESH_TRUNC);  //改成binary和TOZERO玩玩


            // 这里改成9*9了  不然白边太明显了  尺寸越大，建议blursize越大
            Imgproc.GaussianBlur(mskmat,mskmat,new Size(3,3),0);
            Imgproc.resize(mskmat,mskmat, new Size(outWidth,outHeight));
            Imgproc.cvtColor(mskmat,mskmat,Imgproc.COLOR_GRAY2BGR);


            Utils.bitmapToMat(imageBitmap, fg);
            Utils.bitmapToMat(background, bg);

            Imgproc.cvtColor(fg,fg, COLOR_BGRA2BGR);
            Imgproc.cvtColor(bg,bg, COLOR_BGRA2BGR);

            Log.d("FG", String.valueOf(fg.type()));
            Log.d("BG", String.valueOf(bg.type()));


            fg.convertTo(fg,CV_32FC3,1.0/255.0);
            bg.convertTo(bg,CV_32FC3,1.0/255.0);

            Core.subtract(invmskmat,mskmat,invmskmat);
            Core.multiply(fg,invmskmat,fg);
            Core.multiply(bg,mskmat,bg);



            //定义一个resmat存储拼接后的mat
            Mat resmat = new Mat(outHeight, outWidth,CV_32FC3);

            Core.add(bg,fg,resmat);
            //fg.convertTo(fg,CV_8UC3,255);
            resmat.convertTo(resmat,CV_8UC3,255);





//            Imgproc.resize(fg,fg, new Size(previewWidth1,previewHeight1));

//            //再定义一个输出bitmap，就是resbmp，用于存储抠图后fg的输出   previewWidth,previewHeight
             Bitmap resbmp = Bitmap.createBitmap(outWidth,outHeight, Bitmap.Config.ARGB_8888);
//
//            long resize_time_end = System.currentTimeMillis();
//            long resize_time = resize_time_end - resize_time_begin;
//            Log.i("wangqian", "resize_time：" + resize_time);
            Utils.matToBitmap(resmat,resbmp);

            long houchuli_end = System.currentTimeMillis();
            //<--------------------------6.14------------------->


            long houchuli_costTime = (houchuli_end - houchuli_start);
            long end = System.currentTimeMillis();
            long costTime = (end - start);

            Log.i("zhangmingjian", "total_costtime" + costTime);
            Log.i("zhangmingjian", "houchuli_costtime" + houchuli_costTime);


            image.close();
            emitter.onNext(new Result(time, resbmp,outHeight,outWidth));
        }).subscribeOn(Schedulers.io()) // 这里定义被观察者,也就是上面代码的线程, 如果没定义就是主线程同步, 非异步
                // 这里就是回到主线程, 观察者接受到emitter发送的数据进行处理
                .observeOn(AndroidSchedulers.mainThread())
                // 这里就是回到主线程处理子线程的回调数据.
                .subscribe((Result result) -> {
                    boxLabelCanvas.setImageBitmap(result.bitmap);
                    frameSizeTextView.setText(result.outHeight + "x" + result.outWidth);
                    inferenceTimeTextView.setText(Long.toString(result.costTime) + "ms");
                });

    }
}
