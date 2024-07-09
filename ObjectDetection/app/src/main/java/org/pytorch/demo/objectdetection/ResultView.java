package org.pytorch.demo.objectdetection;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;

public class ResultView extends View {

    private final static int TEXT_X = 40;
    private final static int TEXT_Y = 35;
    private final static int TEXT_WIDTH = 260;
    private final static int TEXT_HEIGHT = 50;

    private Paint mPaintRectangle;
    private Paint mPaintText;
    private ArrayList<Result> mResults;

    public ResultView(Context context) {
        super(context);
        init();
    }

    public ResultView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        mPaintRectangle = new Paint();
        mPaintRectangle.setColor(Color.YELLOW);
        mPaintRectangle.setStyle(Paint.Style.STROKE);
        mPaintRectangle.setStrokeWidth(5);

        mPaintText = new Paint();
        mPaintText.setColor(Color.WHITE);
        mPaintText.setStyle(Paint.Style.FILL);
        mPaintText.setTextSize(32);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (mResults != null) {
            for (Result result : mResults) {
                canvas.drawRect(result.rect, mPaintRectangle);

                Path mPath = new Path();
                RectF mRectF = new RectF(result.rect.left, result.rect.top, result.rect.left + TEXT_WIDTH, result.rect.top + TEXT_HEIGHT);
                mPath.addRect(mRectF, Path.Direction.CW);
                mPaintText.setColor(Color.MAGENTA);
                canvas.drawPath(mPath, mPaintText);

                mPaintText.setColor(Color.WHITE);
                canvas.drawText(String.format("%s %.2f", PrePostProcessor.mClasses[result.classIndex], result.score), result.rect.left + TEXT_X, result.rect.top + TEXT_Y, mPaintText);
            }
        }
    }

    public void setResults(ArrayList<Result> results) {
        mResults = results;
        invalidate();
    }
}
