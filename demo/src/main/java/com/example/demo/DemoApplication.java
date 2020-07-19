package com.example.demo;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.json.JSONObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.*;
import java.lang.reflect.*;
import java.util.*;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

@SpringBootApplication
public class DemoApplication implements CommandLineRunner {
	int sum = 0;
	//Lock lock = new ReentrantLock();

	static final String words = "    "; // 避免到处硬编码

	@Autowired
	Calc calc;

	@Autowired // 自动寻找Seller接口的实现并实例化对象
	@Qualifier("waterSeller") //配合Autowired一起，指定使用WineSeller类
	Seller seller;
	@Autowired
	EmployeeSB employeeSB;

	public static void main(String[] args) {
		SpringApplication.run(DemoApplication.class, args);
	}

	@Override
	public void run(String[] args) throws Exception {
		//testLinked(); //线性结构处理
		//testMap();	//非线性结构-字典
		//testSet();	//非线性结构-集合
		//testStr();
		//testTrim();
		//testStringBuilder();

		//testFileRead();
		//testFileWrite();
		//getMaxFreqWord();
		//printRelations();

		//handlerError();

		// Collections
		//Utils.test();
		//testMyComparator();
		//testComparator();
		//testShuffle();
		//testSwap();
		//testFill();
		//testMaxMin();
		//testSearch();

		// Arrays
		//testArray();
		//testArraystoList();

		// Stream
		//testStreamMap();
		//testCreateStream();
		//testCollect();
		//testStreamReduce();

		//testEmployee();
		//testStrStream();

		//testLambda();
		//testLambdaSort();

		// Thread
		//testRunable();
		//testThreadList();
		//testAccountThread();

		// Reflect 反射
		//testReflect();
		//testPrint(args);
		//testAnotation(); // 注解
		//testStaticAgency(); // 静态代理
		//testDynamicAgency(); // 动态代理，基于JDK或CGLIB
		//testAOP(); // 面向切面编程，业务层无感知
		//testCopyFieldByName(); // 根据属性名字，复制不同对象的值
		//testLoadCVSFile(); // 利用CSV文本，动态创建对象

		// 序列化与反序列化
		//testSerializable();
		//testSerializeArrays();

		// Json与XML文件解析
		//useJsonObject();
		//useJackson();
		//testParseXML();

		// 泛型
		//testTemplateClass();
		//testTemplateInterface();
		//testTemplateMethod();

		//SpringBoot
		//testSpringBoot();
		//testEmployeeSB();
		//buildBO().sell();

		//model
		//Singleton singleton = Singleton.getInstance(); //单例模式
		//testMessageCenter(); //观察者模式
		//testFillMethod(); //装饰器模式
		//testProducerConsumer(); //生产者消费者模式
	}

	private void testProducerConsumer() {
		Queue<String> queue = new LinkedList<>();
		List<Thread>  producers = new ArrayList<>();
		List<Thread> consumers = new ArrayList<>();
		for (int i = 0; i < 3; ++i) {
			consumers.add(new Consumer("c-" + i, queue));
		}
		for (Thread thread : consumers) {
			thread.start();
		}
		for (int i = 0; i < 5; ++i) {
			producers.add(new Producer("p-" + i, queue));
		}
		for (Thread thread : producers) {
			thread.start();
		}
	}

	private void testMessageCenter() {
		MessageCenter messageCenter = new MessageCenter();
		Subscriber s1 = new Subscriber("s1");
		Subscriber s2 = new Subscriber("s2");
		Subscriber s3 = new Subscriber("s3");
		s1.register(messageCenter);
		s2.register(messageCenter);
		s3.register(messageCenter);
		messageCenter.notify("msg1");
		s3.unregister(messageCenter);
		Subscriber s4 = new Subscriber("s4");
		s4.register(messageCenter);
		messageCenter.notify("msg2");
	}

	// 依赖注入
	private BO buildBO() {
		BO bo = new BO();
		Map<String, Class> map = new HashMap<>();
		map.put("com.example.demo.Seller", com.example.demo.WineSeller.class);
		Field[] fields = bo.getClass().getDeclaredFields();
		for (Field field : fields) {
			try {
				String typeName = field.getType().getName();
				if (map.containsKey(typeName)) {
					Class clazz = map.get(typeName);
					field.setAccessible(true);
					field.set(bo, clazz.getDeclaredConstructor().newInstance());
				}
			} catch (Exception e) {
				System.out.println("BO exception: " + e);
			}
		}
		return bo;
	}

	private void testEmployeeSB() {
		//employeeSB.setName("Caroline");
		//employeeSB.setGender("Female");
		System.out.println(employeeSB.getName() + ": " + employeeSB.getGender());
	}

	private void testSpringBoot() {
		seller.sell();
	}

	private void testTemplateMethod() {
		//String str = Utils.<String>getFirst(new String[]{"xyz", "abc", "def"});
		String str = Utils.getFirst(new String[]{"xyz", "abc", "def"});
		System.out.println(str);

		//Utils.<DocumentTemplate>print(new Word());
		Utils.print(new Word());
	}

	// 找不到类？？？
	private void testTemplateInterface() {
		Factory<Audi> audiFactory = new CarFactory<>();
		Audi audi = audiFactory.build(Audi.class);
		audi.display();
	}

	private void testTemplateClass() {
		MyPair<String, Integer> pair = new MyPair<>("AAA", 100);
		System.out.println(pair);
		pair.setValue(200);
		System.out.println(pair);

		System.out.println("--------------------");
		List<String> stringArrayList = new ArrayList<String>();
		List<Integer> integerArrayList = new ArrayList<Integer>();
		Class classString = stringArrayList.getClass();
		Class classInteger = integerArrayList.getClass();
		System.out.println(classString);
		if (classString.equals(classInteger)) {
			System.out.println("Same class");
		} else {
			System.out.println("Different class");
		}

		// 不指定模版参数的类型，默认为Object类型
		System.out.println("--------------------");
		MyPair pair2 = new MyPair("ABC", 200);
		System.out.println(pair2);
		pair2.setKey('Z');
		pair2.setValue("XYZ");
		System.out.println(pair2);
	}

	private void testParseXML() {
		parseXML();
		createXML();
	}

	private void parseXML() {
		try {
			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
			DocumentBuilder builder = factory.newDocumentBuilder();
			Document doc = builder.parse("data.xml");
			NodeList nodeList = doc.getElementsByTagName("student");
			displayDomNode(nodeList);
		} catch (Exception e) {
			System.out.println(e.toString());
		}
	}

	private void displayDomNode(NodeList nodeList) {
		for (int i = 0; i < nodeList.getLength(); ++i) {
			Node node = nodeList.item(i);
			NodeList childNodes = node.getChildNodes();
			for (int j = 0; j < childNodes.getLength(); ++j) {
				if (childNodes.item(j).getNodeType() == Node.ELEMENT_NODE) {
					System.out.print(childNodes.item(j).getNodeName() + ": ");
					System.out.println(childNodes.item(j).getFirstChild().getNodeValue());
				}
			}
			System.out.println();
		}
	}

	private void createXML() {
		try {
			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
			DocumentBuilder documentBuilder = factory.newDocumentBuilder();
			Document doc = documentBuilder.newDocument();
			doc.setXmlStandalone(true); // 不调用外部文件
			Element bookstore = doc.createElement("bookstore");
			Element book = doc.createElement("book");
			Element name = doc.createElement("name");
			name.setTextContent("雷神");
			book.appendChild(name);
			book.setAttribute("id", "1");
			bookstore.appendChild(book);
			doc.appendChild(bookstore);

			TransformerFactory transformerFactory = TransformerFactory.newInstance();
			Transformer transformer = transformerFactory.newTransformer();
			StringWriter stringWriter = new StringWriter();
			transformer.transform(new DOMSource(doc), new StreamResult(stringWriter));
			System.out.println(stringWriter.toString());
		} catch (Exception e) {
			System.out.println(e.toString());
		}
	}

	private void useJsonObject() {
		String str = readFile("data.json");
		JSONObject jsonObject = new JSONObject(str);
		Map<String, Object> map = jsonObject.toMap();
		for(Map.Entry<String, Object> entry : map.entrySet()) {
			System.out.println(entry.getKey() + ": " + entry.getValue());
		}
		System.out.println(jsonObject.toString());
	}

	// 为什么抛出异常
	private void useJackson() {
		EmployeeSerial employee = new EmployeeSerial("E001", "Tom", 20, 3500);
		ObjectMapper objectMapper = new ObjectMapper();
		try {
			String jsonString = objectMapper.writeValueAsString(employee);
			System.out.println(jsonString);
			System.out.println("-----------------");
			employee = objectMapper.readValue(jsonString, EmployeeSerial.class);
			System.out.println("employee: " + employee);
		} catch (Exception e) {
			System.out.println("error: " + e.toString());
		}
	}

	private String readFile(String filePath) {
		StringBuilder stringBuilder = new StringBuilder();
		try {
			FileInputStream fileInputStream = new FileInputStream(filePath);
			InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream);
			BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
			String line = "";
			while ((line = bufferedReader.readLine()) != null) {
				stringBuilder.append(line + "\n");
			}
			bufferedReader.close();
		} catch(Exception e) {
			System.out.println(e.toString());
		}
		if (stringBuilder.length() > 0) {
			stringBuilder.deleteCharAt(stringBuilder.length() - 1);
		}
		return stringBuilder.toString();
	}

	private void testSerializeArrays() throws IOException, ClassNotFoundException {
		List<EmployeeSerial> employees = new ArrayList<>();
		employees.add(new EmployeeSerial("E00001", "Tom", 30, 5000));
		employees.add(new EmployeeSerial("E00002", "Jack", 24, 6000));
		employees.add(new EmployeeSerial("E00003", "Mary", 29, 4800));
		employees.add(new EmployeeSerial("E00004", "Jerry", 32, 3900));
		ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
		ObjectOutputStream objectOutputStream = new ObjectOutputStream(byteArrayOutputStream);
		objectOutputStream.writeObject(employees);
		objectOutputStream.close();

		byte[] bytes = byteArrayOutputStream.toByteArray();
		System.out.println("Serialize done");
		ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(bytes);
		ObjectInputStream objectInputStream = new ObjectInputStream(byteArrayInputStream);
		employees = (List<EmployeeSerial>)objectInputStream.readObject();
		System.out.println(employees);
	}

	private void testSerializable() throws IOException, ClassNotFoundException {
		List<EmployeeSerial> employees = new ArrayList<>();
		employees.add(new EmployeeSerial("E00001", "Tom", 30, 5000));
		employees.add(new EmployeeSerial("E00002", "Jack", 24, 6000));
		employees.add(new EmployeeSerial("E00003", "Mary", 29, 4800));
		employees.add(new EmployeeSerial("E00004", "Jerry", 32, 3900));
		for (EmployeeSerial employee : employees) {
			serializeEmployee(employee);
		}
		serializeEmployee(null);
		employees = deserializeEmployee();
		System.out.println(employees);
	}

	private static void serializeEmployee(EmployeeSerial employee) throws FileNotFoundException, IOException {
		String filePath = "employees.dat";
		File file = new File(filePath);
		FileOutputStream fileOutputStream;
		ObjectOutputStream objectOutputStream;
		if (file.exists()) {
			System.out.println("Using existing file");
			fileOutputStream = new FileOutputStream(file, true);
			// 除了第一次写，之后的操作不用写头部信息
			objectOutputStream = new MyObjectOutputStream(fileOutputStream);
		} else {
			System.out.println("Create new file");
			fileOutputStream = new FileOutputStream(file);
			objectOutputStream = new ObjectOutputStream(fileOutputStream);
		}
		objectOutputStream.writeObject(employee);
		objectOutputStream.close();
	}

	private static List<EmployeeSerial> deserializeEmployee() throws FileNotFoundException, IOException, ClassNotFoundException {
		List<EmployeeSerial> employees = new ArrayList<>();
		String filePath = "employees.dat";
		File file = new File(filePath);
		FileInputStream fileInputStream = new FileInputStream(file);
		ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
		EmployeeSerial employee;
		while ((employee = (EmployeeSerial) objectInputStream.readObject()) != null) {
			System.out.println(employee);
			employees.add(employee);
		}
		return employees;
	}

	private void testLoadCVSFile() throws Exception {
		List<Student> students = new ArrayList<>();
		List<Teacher> teachers = new ArrayList<>();
		String filePath = "data.csv";
		FileInputStream fileInputStream = new FileInputStream(filePath);
		InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream);
		BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
		String line;
		while ((line = bufferedReader.readLine()) != null) {
			//System.out.println(line);
			String[] parts = line.split(":");
			if (parts.length == 2) {
				Object object = buildObject(parts[0], parts[1]);
				if (object instanceof Student) {
					students.add((Student)object);
				} else {
					teachers.add((Teacher)object);
				}
			}
		}
		System.out.println("Students: " + students);
		System.out.println("Teachers: " + teachers);
	}

	private Object buildObject(String name, String line) {
		Object object = null;
		try {
			Class clazz = Class.forName("com.example.demo." + name);
			object = clazz.getDeclaredConstructor().newInstance();
			Map<String, Field> fieldMap = buildFieldMap(clazz);
			String[] kvs = line.split(",");
			for (String kv : kvs) {
				String[] pair = kv.split("=");
				//System.out.println(pair[0] + ", " + pair[1]);
				if (fieldMap.containsKey(pair[0])) {
					Field field = fieldMap.get(pair[0]);
					if (field.getType() == Integer.class) {
						field.set(object, Integer.valueOf(pair[1]));
					} else {
						field.set(object, pair[1]);
					}
				}
			}
		} catch (Exception e) {
			System.out.println(e.toString());
		}
		return object;
	}

	private Map<String, Field> buildFieldMap(Class<?> clazz) {
		Map<String, Field> map = new HashMap<>();
		for (Field field : clazz.getDeclaredFields()) {
			int modifiers = field.getModifiers();
			if ((Modifier.isStatic(modifiers)) || (Modifier.isFinal(modifiers))) {
				continue;
			}
			field.setAccessible(true);
			map.put(field.getName(), field);
		}
		return map;
	}

	private void testCopyFieldByName() {
		BoA boA = new BoA("ABC", "BCD", "CDE");
		BoB boB = new BoB();
		Utils.copyByName(boA, boB);
		System.out.println(boA);
		System.out.println(boB);
	}

	// AOP 面向切面编程
	private void testAOP() {
		calc.div(100, 33);
	}

	// 动态代理
	private void testDynamicAgency() {
		SellWine sellMaotai = new SellMaotai();
		InvocationHandler invocationHandler = new NewSell(sellMaotai);
		// newProxyInstance负责创建一个对象，参数分别是：
		// - ClassLoader loader 被代理对象的ClassLoader
		// - Class<?>[] interfaces 接口代表
		// - InvocationHandler h 负责执行代理的对象
		SellWine sellWine = (SellWine)Proxy.newProxyInstance(SellMaotai.class.getClassLoader(),
				SellMaotai.class.getInterfaces(),
				invocationHandler);
		sellWine.sell();

		System.out.println("--------------------");
		SellWine sellMaotai1 = new SellMaotai();
		InvocationHandler invocationHandler1 = new SellWineProxy(sellMaotai1);
		SellWine sellWine1 = (SellWine)Proxy.newProxyInstance(SellMaotai.class.getClassLoader(),
				SellMaotai.class.getInterfaces(),
				invocationHandler1);
		sellWine1.sell();

	}

	// 静态代理
	private void testStaticAgency() {
		SellWine sellMaotai = new SellMaotai();
		NewSellWine newSellWine = new NewSellWine();
		sellMaotai.sell();
		newSellWine.sell(sellMaotai);
	}

	private void testAnotation() throws Exception {
		Class clazz = Class.forName("com.example.demo.MyClass");
		if (clazz.isAnnotationPresent(MyAnotation.class)) {
			MyAnotation anotation = (MyAnotation) clazz.getAnnotation(MyAnotation.class);
			System.out.println(clazz.getName() + ": " + anotation.name() + ", " + anotation.value());
		}

		//获取属性的注解信息
		Field[] fields = clazz.getDeclaredFields();
		for (Field field : fields) {
			if (field.isAnnotationPresent(MyAnotation.class)) {
				MyAnotation anotation = field.getAnnotation(MyAnotation.class);
				System.out.println(field.getName() + ": " + anotation.name() + ", " + anotation.value());
			}
		}

		//获取方法的注解信息
		Method[] methods = clazz.getDeclaredMethods();
		for (Method method : methods) {
			if (method.isAnnotationPresent(MyAnotation.class)) {
				MyAnotation anotation = method.getAnnotation(MyAnotation.class);
				System.out.println(method.getName() + ": " + anotation.name() + ", " + anotation.value());
			}
		}
	}

	private void testPrint(String[] args) throws Exception {
		String name;
		//System.out.println(EmployeeSB.class.getName());
		if (args.length > 0) {
			name = args[0];
		} else {
			Scanner in = new Scanner(System.in);
			System.out.println("Enter class name (e.g. java.util.Date): ");
			name = in.next();
		}

		try {
			Class clazz = Class.forName(name);
			Class superClass = clazz.getSuperclass();
			String modifies = Modifier.toString(clazz.getModifiers()); // 找修饰符
			if (modifies.length() > 0) {
				System.out.print(modifies + " ");
			}
			System.out.print("class " + name);
			if ((superClass != null) && (superClass != Object.class)) { // 所有的类都从Object继承下来
				System.out.print(" extends " + superClass.getName());
			}
			System.out.print(" { \n");
			printFields(clazz); 		// 打印属性
			System.out.println();
			printConstructors(clazz); 	// 打印构造函数
			System.out.println();
			printMethods(clazz); 		// 打印方法
			System.out.print("}\n");
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		}
		System.exit(0);
	}

	public static void printFields(Class clazz) {
		Field[] fields = clazz.getDeclaredFields();
		for (Field field : fields) {
			System.out.print(words);
			String modifies = Modifier.toString(field.getModifiers());
			if (modifies.length() > 0) {
				System.out.print(modifies + " ");
			}
			String type = field.getType().getName();
			String name = field.getName();
			System.out.print(type + " " + name + ";\n");
		}
	}

	public static void printConstructors(Class clazz) {
		Constructor[] constructors = clazz.getDeclaredConstructors();
		for (Constructor constructor : constructors) {
			System.out.print(words);
			String modifies = Modifier.toString(constructor.getModifiers());
			if (modifies.length() > 0) {
				System.out.print(modifies + " ");
			}
			String name = constructor.getName();
			System.out.print(name + "(");
			printMethodParameters(constructor);
			System.out.print(");\n");
		}
	}

	public static void printMethods(Class clazz) {
		Method[] methods = clazz.getDeclaredMethods();
		for (Method method : methods) {
			System.out.print(words);
			String modifies = Modifier.toString(method.getModifiers());
			if (modifies.length() > 0) {
				System.out.print(modifies + " ");
			}
			String type = method.getReturnType().getName();
			String name = method.getName();
			System.out.print(type + " " + name + "(");
			printMethodParameters(method);
			System.out.print(");\n");
		}
	}

	public static void printMethodParameters(Executable exe) {
		Class[] paramTypes = exe.getParameterTypes();
		for (int i = 0; i < paramTypes.length; ++i) {
			if (i > 0) {
				System.out.print(", ");
			}
			System.out.print(paramTypes[i].getName());
		}
	}

	private void testReflect() throws Exception {
		Class clazz = Class.forName("com.example.demo.Product");
		Constructor constructor = clazz.getDeclaredConstructor(String.class, Integer.class, Float.class);
		Product product = (Product)constructor.newInstance("Nike", 12345, 788.99f); // 动态的构造一个类
		product.showInfo();

		// 更新属性
		Method setBrand = clazz.getMethod("setBrand", String.class);
		Method setId = clazz.getMethod("setId", Integer.class);
		Method setPrice = clazz.getMethod("setPrice", Float.class);
		setBrand.invoke(product, "LiNing");
		setId.invoke(product, 23456);
		setPrice.invoke(product, 577.99f);
		product.showInfo();

		Field price = clazz.getDeclaredField("price");
		price.setAccessible(true);
		price.set(product, 288.99f);
		product.showInfo();
	}

	private void testAccountThread() {
		Account account = new Account();
		List<Thread> threadList = new ArrayList<>();
		for (int i = 0; i < 100; ++i) {
			Thread t;
			if ((i % 2) == 0) {
				/*
				//匿名类
				t = new Thread(new Runnable() {
					@Override
					public void run() {
						synchronized (account) {
							for (int i = 0; i < 10000; ++i) {
								account.deposit(i);
							}
						}
					}
				}, "T-" + i);
				 */
				// Lambda表达式
				t = new Thread(() -> {
					synchronized (account) {
						for (int j = 0; j < 10000; ++j) {
							account.deposit(j);
						}
					}
				}, "T-" + i);
			} else {
				t = new Thread(new Runnable() {
					@Override
					public void run() {
						synchronized (account) {
							for (int i = 0; i < 10000; ++i) {
								account.withdraw(i);
							}
						}
					}
				}, "T-" + i);
			}
			threadList.add(t);
		}
		for (Thread thread : threadList) {
			thread.start();
		}
		for (Thread thread : threadList) {
			try {
				thread.join();
			} catch (InterruptedException e) {
				System.out.println("ThreadList interrupted");
			}
		}
		System.out.println("Account balance is " + account.getBalance());
	}

	private void testThreadList() {
		MathOperationClass mathOperation = new MathOperationClass();
		List<Thread> threadList = new ArrayList<>();
		for (int i = 0; i < 100; ++i) {
			Thread t;
			if ((i % 2) == 0) {
				t = new Thread(new Runnable() {
					@Override
					public void run() {
						//lock.lock();
						//lock.tryLock();
						// 拿到mathOperation才可以执行，没有拿到只能等，以保证顺序执行
						synchronized (mathOperation) {
							for (int i = 0; i < 10000; ++i) {
								sum = mathOperation.add(sum, i);
							}
						}
						//lock.unlock();
					}
				}, "T-" + i);
			} else {
				t = new Thread(new Runnable() {
					@Override
					public void run() {
						//lock.lock();
						synchronized (mathOperation) {
							for (int i = 0; i < 10000; ++i) {
								sum = mathOperation.minus(sum, i);
							}
						}
						//lock.unlock();
					}
				}, "T-" + i);
			}
			threadList.add(t);
		}
		for (Thread thread : threadList) {
			thread.start();
		}
		// 主线程启动后，不使用join，程序先去print了，有可能程序还没走完，print拿到的是中间结果
		for (Thread thread : threadList) {
			try {
				// 等到所有线程执行结束
				thread.join();
			} catch (InterruptedException e) {
				System.out.println("ThreadList interrupted");
			}
		}
		System.out.println(sum);
		System.out.println(sum);
		System.out.println(sum);
		System.out.println(sum);
		System.out.println(sum);
	}

	private void testRunable() {
		MyRunable r1 = new MyRunable("T-1");
		MyRunable r2 = new MyRunable("T-2");
		r1.start();
		r2.start();
		r1.join(); // 确保主线程在结束之前不会被意外退出
		r2.join();
	}

	private void testLambdaSort() {
		List<Integer> list = new ArrayList<>();
		Collections.addAll(list, 1, 9, 8, 6, 7, 5, 4, 2, 3);
		/*
		// Lambda表达式
		Collections.sort(list, (x, y) -> {
			if (x == y) {
				return 0;
			} else if (x > y) {
				return -1;
			} else {
				return 1;
			}
		});
		*/
		// 匿名类，接口有多少方法都要实现，一个都不能少
		Collections.sort(list, new Comparator<Integer>() {
			@Override
			public int compare(Integer o1, Integer o2) {
				if(o1 == o2) {
					return 0;
				} else if (o1 > o2) {
					return -1;
				} else {
					return 1;
				}
			}
		});
		System.out.println(list);
	}

	private void testLambda() {
		MathOperation add = (int x, int y) -> {return x + y;};
		MathOperation minus = (int x, int y) -> {return x - y; };
		System.out.println(op(100, 200, add));
		System.out.println(op(100, 200, minus));
		System.out.println(op(100, 200, (x, y) -> x * y));
		// 匿名类
		System.out.println(op(100, 200, new MathOperation() {
			@Override
			public int operation(int a, int b) {
				return a * b;
			}
		}));
	}

	private int op(int a, int b, MathOperation mathOperation) {
		return mathOperation.operation(a, b);
	}

	private void testStrStream() {
		String[] names = new String[]{"abcd", "12", "345", "1234", "DEFG"};
		long count = Arrays.stream(names).filter(str -> (str.length() > 3)).count();
		System.out.println(count);
	}

	private void testEmployee() {
		List<Employee> employees = new ArrayList<>();
		employees.add(new Employee("Mike", 8000));
		employees.add(new Employee("Tom", 12000));
		employees.add(new Employee("Lucy", 3000));
		employees.add(new Employee("Marry", 4000));
		Stream<Employee> employeeStream = employees.stream();
		List<Employee> salary5000 = employeeStream
				.filter(e -> (e.getSalary() > 5000))
				.collect(Collectors.toList());
		System.out.println(salary5000);

		employeeStream = employees.stream();
		Integer totalSalary = employeeStream
				.map(e -> e.getSalary())
				.reduce(0, Integer::sum);
		System.out.println(totalSalary);
	}

	private void testStreamReduce() {
		List<Integer> list = Arrays.asList(1, 2, 3, 4, 5);
		Integer res = list.stream().reduce(0, (x, y) -> (x + y));
		System.out.println(res);

		List<Integer> list2 = Arrays.asList();
		Optional<Integer> res2 = list2.stream().reduce((x, y) -> (x + y));
		System.out.println(res2.isPresent());

		List<Integer> list3 = Arrays.asList(1, 2, 3);
		Optional<Integer> res3 = list3.stream().reduce((x, y) -> (x + y));
		System.out.println(res3.isPresent());
		System.out.println(res3.get());

		List<Integer> list4 = Arrays.asList(1, 3, 2);
		Integer max = list4.stream().reduce(0, Integer::max);
		System.out.println(max);
	}

	private void testCollect() {
		List<String> strings = Arrays.asList("abc", "", "bc", "efg", "bcd", "", "jkl");
		List<String> filtered = strings.stream()
				.filter(str -> !str.isEmpty())
				.collect(Collectors.toList());
		System.out.println(filtered);

		String newStrings = strings.stream()
				.filter(str -> !str.isEmpty())
				.collect(Collectors.joining(","));
		System.out.println(newStrings);
	}

	private void testCreateStream() {
		int[] array = new int[]{0, 3, 2};
		IntStream inStream = Arrays.stream(array);
//		inStream.forEach(System.out::println);
		inStream.forEachOrdered(System.out::println); //对有序容器，按照输入顺序，有序打印

		Employee[] employees = new Employee[]{
				new Employee("Mike", 3000),
				new Employee("jack", 2000),
				new Employee("Tom", 8000)
		};
		Stream<Employee> stream = Arrays.stream(employees);
//		stream.forEach(System.out::println);
		stream.forEachOrdered(System.out::println);
	}

	private void testStreamMap() {
		List<Integer> list = new ArrayList<>();
		Collections.addAll(list, 0, 1, 2, 3, 4, 5, 6);
		List<Integer> squares = list.stream()
				.map(i -> i * i)
				.collect(Collectors.toList());
		System.out.println(squares);
	}

	private void testArraystoList() {
		Integer[] array = new Integer[]{1, 2, 3};
		List<Integer> list = Arrays.asList(array);
		System.out.println(list.toString());
	}

	private void testArray() {
		Integer[] array = new Integer[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
		Arrays.sort(array);
		System.out.println(Arrays.toString(array));
//		for (Integer i : array) {
//			System.out.printf("%d ", i);
//		}
//		System.out.println();

		Arrays.sort(array, new MyComparator());
		System.out.println(Arrays.toString(array));
//		for (Integer i : array) {
//			System.out.printf("%d ", i);
//		}
//		System.out.println();

		Employee[] employees = new Employee[]{
				new Employee("Mike", 3000),
				new Employee("jack", 2000),
				new Employee("Tom", 8000)
		};
		Arrays.sort(employees); // 采用自定义的类、排序方法
		System.out.println(Arrays.toString(employees));
	}

	// 有序数组，查找效率 O(log2n)
	// binarySearch 必须传入有序数组
	private void testSearch() {
		List<Integer> list = new ArrayList<>();
		Collections.addAll(list, 0,1,2,6,4,3,5,7,8,9);
		System.out.println(Collections.binarySearch(list, 3));
		System.out.println(Collections.binarySearch(list, 10));
		System.out.println(list);
	}

	private void testMaxMin() {
		List<Integer> list = new ArrayList<>();
		Collections.addAll(list, 0,1,2,3,4,5,6,7,8,9);
		System.out.println(Collections.max(list));
		System.out.println(Collections.min(list));
		System.out.println(Collections.max(list, new MyComparator()));
		System.out.println(Collections.min(list, new MyComparator()));
	}

	private void testFill() {
		List<Integer> list = new ArrayList<>();
		Collections.addAll(list, 0,1,2,3,4,5,6,7,8,9);
		Collections.fill(list, 0);
		System.out.println(list);
	}

	private void testSwap() {
		List<Integer> list = new ArrayList<>();
		Collections.addAll(list, 0,1,2,3,4,5,6,7,8,9);
		// swap
//		Integer i = list.get(3);
//		list.set(3, list.get(5));
//		list.set(5, i);
		Collections.swap(list, 3, 5);
		System.out.println(list);

	}

	private void testShuffle() {
		List<Integer> list = new ArrayList<>();
//		for (int i = 0; i < 10; ++i) {
//			list.add(i);
//		}
		Collections.addAll(list, 0,1,2,3,4,5,6,7,8,9);
		Collections.shuffle(list);
		System.out.println(list);
	}

	private void testComparator() {
		List<Employee> employees = new ArrayList<>();
		employees.add(new Employee("Tom", 3000));
		employees.add(new Employee("Mike", 8000));
		employees.add(new Employee("Jerry", 6500));
		employees.add(new Employee("Marry", 9800));
		Collections.sort(employees);
		System.out.println(employees);
		Collections.reverse(employees);
		System.out.println(employees);
	}

	private void testMyComparator() {
		List<Integer> list = new ArrayList<>();
		list.add(10);
		list.add(7);
		list.add(5);
		list.add(8);
		Collections.sort(list);
		System.out.println(list);
		Collections.sort(list, new MyComparator());
		System.out.println(list);
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

	private void testTrim() {
		String s = "     a   \t  \t ";  //"   abcde\t bcd";
		String ts = trim(s);
		System.out.println(s.length());
		System.out.println(ts.length());
		System.out.println(ts);
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
