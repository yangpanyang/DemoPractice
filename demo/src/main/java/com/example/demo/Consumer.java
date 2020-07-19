package com.example.demo;

import java.util.Queue;

public class Consumer extends Thread {
    String name;
    Queue<String> queue;

    Consumer(String name, Queue<String> queue) {
        this.name = name;
        this.queue = queue;
    }
}
