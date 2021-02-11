package com.example.demo;

public interface Factory<T> {
    public T build(Class<T> clazz);
}
