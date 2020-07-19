package com.example.demo;

import java.util.Queue;

public class Producer extends Thread {
    String name;
    Queue<String> queue;

    Producer(String name, Queue<String> queue) {
        this.name = name;
        this.queue = queue;
    }

    public void produce() {

    }
}
