package com.example.demo;

public interface Factory<T> {
    // 生产T类型的对象：根据类型动态的生成一个对象
    public T build(Class<T> clazz);
}
