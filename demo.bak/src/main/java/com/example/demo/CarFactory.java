package com.example.demo;

public class CarFactory<T> implements Factory<T> {
    @Override
    public T build(Class<T> clazz) {
        T object = null;
        try {
            object = (T)clazz.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            System.out.println(e.toString());
        }
        return null;
    }
}
