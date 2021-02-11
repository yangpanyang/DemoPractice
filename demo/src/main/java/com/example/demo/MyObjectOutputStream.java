package com.example.demo;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;

// 用于处理连续读取对象的情况
public class MyObjectOutputStream extends ObjectOutputStream {
    public MyObjectOutputStream() throws IOException {
        super();
    }

    public MyObjectOutputStream(OutputStream out) throws IOException {
        super(out);
    }

    @Override
    protected void writeStreamHeader() throws IOException {
        return;  // 想要连续写数据，第一次读了header，之后不用再读
    }
}
