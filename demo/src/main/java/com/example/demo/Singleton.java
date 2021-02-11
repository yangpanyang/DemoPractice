package com.example.demo;

// 单例模式
public class Singleton {
    private String message;
//    private static Singleton singleton = null;
//    // 多线程的时候，会造成instance不唯一，两种改进方式如下
//    public static Singleton getInstance() {
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

    // 这个实例是属于这个类的，只有一个实例
    private static Singleton singleton = null;
    // synchronized保证有竞争的情况下，构造的对象也是唯一的
    public static synchronized Singleton getInstance() {
        if (singleton == null) {
            singleton = new Singleton();
        }
        return singleton;
    }

    public void doSomething() {
        System.out.println(message);
    }

    // 构造函数私有化，外面的类不可以直接访问到
    private Singleton() {
        System.out.println("Create instance");
    }
}
