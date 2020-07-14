package com.example.demo;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.*;
import java.util.*;

@SpringBootApplication
public class DemoApplication implements CommandLineRunner {

	public static void main(String[] args) {
		SpringApplication.run(DemoApplication.class, args);
	}

	@Override
	public void run(String[] args) throws Exception {
		//testLinked(); //线性结构处理
		//testMap();	//非线性结构-字典
		//testSet();	//非线性结构-集合
		//testStr();

//		String s = "     a   \t  \t ";//"   abcde\t bcd";
//		String ts = trim(s);
//		System.out.println(s.length());
//		System.out.println(ts.length());
//		System.out.println(ts);

		//testStringBuilder();

		//testFileRead();
		//testFileWrite();
		//getMaxFreqWord();
		//printRelations();

		handlerError();
	}

	private void handlerError() throws Exception {
		try {
//			int x = 100, y = 0;
//			System.out.println(x / y);
			throwMyException("handlerError");
		} catch (MyException e) {
			System.out.println(e.toString());
		} finally {
			System.out.println("finally");
		}
	}

	private void throwMyException(String str) throws MyException {
		throw new MyException(str);
	}

	private void printRelations() throws Exception {
		String filePath = "friends.txt";
		FileInputStream fileInputStream = new FileInputStream(filePath);
		InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream);
		BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
		String line = "";
		HashMap<String, List<String>> friends = new HashMap<>();
		while ((line = bufferedReader.readLine()) != null) {
			line = line.toLowerCase();
			String[] names = line.split(" ");
			if (!friends.containsKey(names[0])) {
				friends.put(names[0], new ArrayList<>());
			}
			friends.get(names[0]).add(names[1]);
		}
		bufferedReader.close();
		for (Map.Entry<String, List<String>> entry: friends.entrySet()) {
			System.out.printf("%s: ",entry.getKey());
			List<String> names = entry.getValue();
			for (int i = 0; i < names.size(); ++i) {
				System.out.printf("%s", names.get(i));
				if (i < (names.size() - 1)) {
					System.out.printf(", ");
				} else {
					System.out.printf("\n");
				}
			}
		}
	}

	private void getMaxFreqWord() throws Exception {
		String filePath = "words.txt";
		FileInputStream fileInputStream = new FileInputStream(filePath);
		InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream);
		BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
		String line, maxFreqWord = "";
		int maxFreq = 0;
		HashMap<String, Integer> count = new HashMap<>();
		while ((line = bufferedReader.readLine()) != null) {
			String[] words = line.split(" ");
			for (String word : words) {
				// *** 统一处理，使代码更简洁 ***
				if (!count.containsKey(word)) {
					count.put(word, 0);
				}
				count.put(word, count.get(word) + 1);
				if (count.get(word) > maxFreq) {
					maxFreq = count.get(word);
					maxFreqWord = word;
				}
			}
		}
		bufferedReader.close();
		System.out.printf("%s appears %d times.\n", maxFreqWord, maxFreq);
	}

	private void testFileWrite() throws Exception {
		String filePath = "output.txt";
		OutputStream outputStream = new FileOutputStream(filePath);
		String s = "Hello, World!\n";
		byte b[] = s.getBytes();
		outputStream.write(b);
		outputStream.write(b);
		outputStream.close();
	}

	private void testFileRead() throws Exception {
		String cwd = System.getProperty("user.dir"); //获取当前工作路径
		System.out.println(cwd);

		String filePath = "pom.xml";
		FileInputStream fileInputStream = new FileInputStream(filePath);
		InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream);
		BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
		String line = "";
		while ((line = bufferedReader.readLine()) != null) {
			System.out.println(line);
		}
		bufferedReader.close();
	}
	
	private void testStringBuilder() {
		StringBuilder sb = new StringBuilder();
		sb.append(7);
		sb.append("天加油");
		System.out.println(sb.toString());
		sb.setCharAt(0, '七');
		sb.append(" 数据结构");
		sb.insert(5, "Java与");
		System.out.println(sb.toString());
	}

	private String trim(String str) { //字符串 头尾去空格
		int beginIndex = 0, endIndex = str.length() - 1;
		while (beginIndex < str.length()) {
			if ((str.charAt(beginIndex) == ' ') || (str.charAt(beginIndex) == '\t')) {
				beginIndex += 1;
			} else {
				break;
			}
		}
		while (endIndex >= beginIndex) {
			if ((str.charAt(endIndex) == ' ') || (str.charAt(endIndex) == '\t')) {
				endIndex -= 1;
			} else {
				break;
			}
		}
		if (beginIndex > endIndex) {
			return "";
		} else {
			return str.substring(beginIndex, endIndex + 1);
		}
	}

	private void testStr() {
		// StringBuilder, StringBuffer 字符数组 可以编辑
		// 从运算速度来说，StringBuilder 最快，StringBuffer 次之(线程安全的)，String 最慢
		String str = "字符串不能编辑，字符数组可以编辑.";
		String str2 = "请用StringBuilder, StringBuffer！";
		System.out.printf("%s\n", str);
		for (int i = 0; i < str.length(); ++i) {
			System.out.printf("%c", str.charAt(i));
		}
		System.out.println();
		System.out.printf("%s\n", str + str2);
	}

	private void testSet() {
		HashSet<String> set = new HashSet<>();
		set.add("ABC");
		set.add("BCD");
		set.add("DEF");
		set.add("ABC");
		for (String key : set) {
			System.out.println(key);
		}
	}

	private void testMap() {
		System.out.println("---------- HashMap ------------");
		HashMap<String, Integer> map = new HashMap();
		map.put("ABC", 100);
		map.put("BCD", 200);
		System.out.println(map.size());
		System.out.println(map.containsKey("CDE"));
		System.out.println(map.containsKey("ABC"));
		System.out.println(map.get("ABC"));
		System.out.println(map.get("DEF"));
		map.put("zzz", null);
		System.out.println(map.get("zzz"));

//		HashMap<String, Integer> map = new HashMap<>();
		map.clear();
		map.put("A", 1);
		map.put("B", 2);
		map.put("C", 3);
		for (String key : map.keySet()) {
			System.out.println(key + ": " + map.get(key));
		}
		for (Map.Entry<String, Integer> entry : map.entrySet()) {
			System.out.println(entry.getKey() + ": " + entry.getValue());
		}
		for (Integer value : map.values()) {
			System.out.println(value);
		}
	}

	private void testLinked() {
		System.out.println("----------- Array -----------");
		Double[] array = new Double[]{123.45, 345.34, 564.3};
		//array[0] = "ABC";
		//array[1] = "BCD";
		System.out.printf("%s\n", array[0]);
		System.out.printf("Array size: %d\n", array.length);
		array = new Double[10];
		System.out.println("Array size: " + array.length);

		System.out.println("---------- ArrayList ------------");
		List<Integer> vector = new ArrayList<>();
		vector.add(0);
		vector.add(1);
		System.out.printf("size: %d\n", vector.size());
		System.out.printf("%d, %d\n", vector.get(0), vector.get(1));

		System.out.println("---------- LinkedList ------------");
		LinkedList<Integer> list = new LinkedList<>();
		list.add(0);
		list.add(1);
		list.add(2);
		list.addFirst(-1);
		list.addLast(-2);
		System.out.printf("Size: %d\n", list.size());
		System.out.printf("%d, %d, %d\n", list.get(0), list.get(1), list.get(4));

		System.out.println("---------- Random ------------");
		Random random = new Random();
		int target = random.nextInt(100);
		Scanner scanner = new Scanner(System.in);
		while (true) {
			int data = scanner.nextInt();
			if (data == target) {
				System.out.println("Pass");
				break;
			} else if (data > target) {
				System.out.println("bigger");
			} else {
				System.out.println("smaller");
			}
		}
	}
}
