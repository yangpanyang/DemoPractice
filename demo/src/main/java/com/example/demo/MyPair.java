package com.example.demo;

// 模版参数T1、T2，在编译的时候T1、T2会变成实际对应的类型
// 泛型只在编译时有效、在运行时并没有这个信息，所有和泛型相关的检查都会在编译的时候做掉
public class MyPair<T1, T2> {
    T1 key;
    T2 value;

    MyPair(T1 key, T2 value) {
        this.key = key;
        this.value = value;
    }

    public void setKey(T1 key) {
        this.key = key;
    }

    public void setValue(T2 value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return key.toString() + ": " + value.toString();
    }
}
