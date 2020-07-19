package com.example.demo;

public class MyRunable implements Runnable {
    private Thread thread;
    private String threadName;

    // 初始化，千万不要相信默认值
    MyRunable (String threadName) {
        this.threadName = threadName;
        thread = null;
        System.out.println("Creating " + threadName);
    }

//    @Override // Runnable并没有实现，是一个纯抽象的函数，可以不用Override
    public void run() {
        System.out.println("Running " + threadName);
        try { // Thread.sleep 会抛出异常，一定要捕捉异常
            for (int i = 0; i < 4; ++i) {
                System.out.println("Thread " + threadName + ", " + i);
                Thread.sleep(200);
            }
        } catch (InterruptedException e) {
            System.out.println("Thread " + threadName + " interrupted");
        }
        System.out.println("Thread " + threadName + " exiting");
    }

    // 每一个Runable跟一个线程绑定
    public void start() {
        System.out.println("Starting " + threadName);
        if (thread == null) {
            thread = new Thread(this, threadName);
        }
        thread.start();
    }

    public void join() {
        if (thread != null) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                System.out.println("Thread interrupted");
            }
        }
    }
}
