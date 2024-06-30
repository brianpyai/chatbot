#include "mydata.h"

void randomStr(char *str, size_t length)
{
	static const char alphanum[] =
		"0123456789"
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"abcdefghijklmnopqrstuvwxyz";
	static const size_t alphanum_size = sizeof(alphanum) - 1;

	for (size_t i = 0; i < length; ++i)
	{
		str[i] = alphanum[rand() % alphanum_size];
	}
	str[length] = '\0';
}

int testPerformance(int ts = 10000 * 5)
{
	printf("Testing Performance for %d Million times:\n", ts / 1000 / 1000);
	clock_t start, finish;
	double total_time;

	char randomPart[33]; // 32个字符 + 结束符
	randomStr(randomPart, 32);
	printf("%s", randomPart);

	// std::unordered_map 测试
	unordered_map<string, string> stdMap;
	string stdKey, stdValue;

	printf("\nTesting std::unordered_map performance:\n");
	// 测试 std::unordered_map insert 性能
	start = clock();
	for (int i = 0; i < ts; i++)
	{
		stdKey = to_string(i) + randomPart;
		stdValue = to_string(i) + randomPart;
		stdMap[stdKey] = stdValue;
	}
	finish = clock();
	total_time = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("std::unordered_map insert: %lf secs, %lf M/sec\n", total_time, ts / total_time / 1000 / 1000);

	// 测试 std::unordered_map find 性能
	start = clock();
	for (int i = 0; i < ts; i++)
	{
		stdKey = to_string(i) + randomPart;
		auto it = stdMap.find(stdKey);
		if (it == stdMap.end())
		{
			printf("Error: key not found in std::unordered_map\n");
			//break;
		}
	}
	finish = clock();
	total_time = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("std::unordered_map find: %lf secs, %lf M/sec\n", total_time, ts / total_time / 1000 / 1000);

	// 添加 myDict 测试
	myDict dict;
	myChar key, value;

	printf("\nTesting myDict performance:\n");
	// 测试 myDict set 性能
	start = clock();
	for (int i = 0; i < ts; i++)
	{
		key.length = 0;	  // 重置 key
		value.length = 0; // 重置 value

		key.cat(to_string(i).c_str());
		key.cat(randomPart);

		// 生成 value: 7位数字 + 32个随机字符
		value.cat(to_string(i).c_str());
		value.cat(randomPart);

		dict.set(key.chars, value.chars);
	}
	finish = clock();
	total_time = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("myDict set: %lf secs, %lf M/sec\n", total_time, ts / total_time / 1000 / 1000);

	// 测试 myDict get 性能
	start = clock();
	for (int i = 0; i < ts; i++)
	{
		key.length = 0; // 重置 key
		value.length = 0;
		// 重新生成相同的 key
		key.cat(to_string(i).c_str());
		key.cat(randomPart);

		const char *result = dict.get(key.chars);
	}
	finish = clock();
	total_time = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("myDict get: %lf secs, %lf M/sec\n", total_time, ts / total_time / 1000 / 1000);

	return 0;
}

int main()
{
	testPerformance(10000 * 500);
	const int dictSize = 3000;
	const int testCount = 1000;
	const int printCount = 10;

	char randomPart[33]; // 32个字符 + 结束符
	randomStr(randomPart, 32);
	printf("Random part: %s\n", randomPart);

	myDict dict;
	myChar key, value;

	// 生成3000个items的myDict
	for (int i = 0; i < dictSize; i++)
	{
		key.length = 0;
		value.length = 0;

		key.cat(to_string(i).c_str());
		key.cat(randomPart);

		value.cat(to_string(i).c_str());
		value.cat(randomPart);

		dict.set(key.chars, value.chars);
	}

	// 生成0到2999的随机序列
	std::vector<int> indices(dictSize);
	for (int i = 0; i < dictSize; i++)
	{
		indices[i] = i;
	}
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(indices.begin(), indices.end(), g);

	// 随机验证100个，并打印其中10个结果
	int printedCount = 0;
	for (int i = 0; i < testCount; i++)
	{
		int index = indices[i];

		key.length = 0;
		value.length = 0;

		key.cat(to_string(index).c_str());
		key.cat(randomPart);

		value.cat(to_string(index).c_str());
		value.cat(randomPart);

		const char *result = dict.get(key.chars);

		bool match = (result && strcmp(result, value.chars) == 0);

		if (printedCount < printCount)
		{
			printf("Test %d:\n", i + 1);
			printf("Key: %s\n", key.chars);
			printf("Expected: %s\n", value.chars);
			printf("Got     : %s\n", result ? result : "(null)");
			printf("Match   : %s\n", match ? "Yes" : "No");
			printf("-------------------\n");
			printedCount++;
		}

		if (!match)
		{
			printf("Error: Mismatch found at index %d\n", index);
		}
	}

	printf("Verification complete. %d items tested.\n", testCount);
	return 0;
}