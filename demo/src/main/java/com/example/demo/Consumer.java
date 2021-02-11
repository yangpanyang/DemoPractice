package com.example.demo;

import java.util.Queue;

public class Consumer extends Thread {
    String name;
    Queue<String> queue;

    Consumer(String name, Queue<String> queue) {
        this.name = name;
        this.queue = queue;
    }

    public void consume() {
        while (true) {
            synchronized (queue) {  // 为了保证在队列里拿东西消费的时候是同步的
                if (!queue.isEmpty()) {  // 主动判断，Kafka消费的时候可以等待不用主动判空
                    String product = queue.poll();
                    System.out.println(name + " consume " + product);
                }
            }
        }
    }

    @Override
    public void run() {
        consume();
    }
}
