package com.example.demo;

// 单例模式
public class Singleton {
    private String message;
//    private static Singleton singleton = null;
//    // 多线程的时候，会造成instance不唯一
//    static Singleton getInstance() {
//        if (singleton == null) {
//            singleton = new Singleton();
//        }
//        return singleton;
//    }

    // static静态成员的初始化，在构造的时候实现，加载的时候会慢一些
//    private static final Singleton singleton = new Singleton();
//    static Singleton getInstance() {
//        return singleton;
//    }

    private static Singleton singleton = null;
    static synchronized Singleton getInstance() {
        if (singleton == null) {
            singleton = new Singleton();
        }
        return singleton;
    }

    public void doSomething() {
        System.out.println(message);
    }

    private Singleton() {
        System.out.println("Create instance");
    }
}
