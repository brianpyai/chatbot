#ifndef MYDATA_H
#define MYDATA_H
#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <ctime>
#include <unordered_map>
#include <sstream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
using namespace std;
#define BUFFER_SIZE 1024 * 64
long sec = 999999999;
int fast_compare(const char *ptr0, const char *ptr1, long long len)
{
	long long fast = len / sizeof(size_t) + 1;
	long long offset = (fast - 1) * sizeof(size_t);
	long long current_block = 0;

	if (len <= sizeof(size_t))
	{
		fast = 0;
	}

	size_t *lptr0 = (size_t *)ptr0;
	size_t *lptr1 = (size_t *)ptr1;

	while (current_block < fast)
	{
		if ((lptr0[current_block] ^ lptr1[current_block]))
		{
			int pos;
			for (pos = current_block * sizeof(size_t); pos < len; ++pos)
			{
				if ((ptr0[pos] ^ ptr1[pos]) || (ptr0[pos] == 0) || (ptr1[pos] == 0))
				{
					return (int)((unsigned char)ptr0[pos] - (unsigned char)ptr1[pos]);
				}
			}
		}

		++current_block;
	}

	while (len > offset)
	{
		if ((ptr0[offset] ^ ptr1[offset]))
		{
			return (int)((unsigned char)ptr0[offset] - (unsigned char)ptr1[offset]);
		}
		++offset;
	}

	return 0;
}

int startsWith(const char *str, const char *prefix)
{
	return fast_compare(str, prefix, strlen(prefix)) == 0;
}
int endsWith(const char *s, const char *t)
{
	size_t slen = strlen(s);
	size_t tlen = strlen(t);
	if (tlen > slen)
		return 0;
	return fast_compare(s + slen - tlen, t, tlen) == 0;
}

char *getCharRange(const char *str, int start, int end)
{
	int length = end - start + 1;
	char *subStr = (char *)malloc((length + 1) * sizeof(char));

	for (int i = start; i <= end; i++)
	{
		subStr[i - start] = str[i];
	}
	subStr[length] = '\0';
	return subStr;
}

struct myChar
{
	std::vector<char *> charsPtr;
	char *chars = NULL;
	long long length = 0;
	long long mSize = 0;
	int buffer_size = 1024 * 256;

	myChar(const char *c = "")
	{
		cat(c, strlen(c));
		charsPtr.push_back(chars);
	}

	~myChar()
	{
		if (chars != NULL)
		{
			free(chars);
			chars = NULL;
		}
		length = 0;
		mSize = 0;
	}

	int startswith(const char *c)
	{
		return startsWith(chars, c);
	}

	int endswith(const char *c)
	{
		return endsWith(chars, c);
	}

	long long find(const char *c) const // 添加 const 关键字
	{
		char *p = strstr(chars, c);
		if (!p)
			return -1;
		return (long long)(p - chars); // 注意：这里移除了 +1
	}

	void cat(const char *c, long long len = 0)
	{
		if (len == 0)
			len = strlen(c);
		long long total_size = length + len + 1;
		if (total_size > mSize)
		{
			total_size += buffer_size;
			chars = (char *)realloc(chars, total_size);
			if (chars == NULL)
			{
				return;
			}

			mSize = total_size;
		}

		memcpy(chars + length, c, len);
		length += len;
		chars[length] = '\0';
	}

	myChar operator*(const int n)
	{
		myChar result;
		for (int i = 0; i < n; i++)
		{
			result.cat(chars, length);
		}
		return result;
	}

	void findAll(const char *c, std::vector<long long> &positions)
	{
		char *p = chars;
		while ((p = strstr(p, c)) != NULL)
		{
			positions.push_back(p - chars);
			p += strlen(c);
		}
	}

	void replace(const char *c, const char *replacement)
	{
		std::vector<long long> positions;
		findAll(c, positions);
		int cLength = strlen(c);
		int rLength = strlen(replacement);
		int diff = rLength - cLength;
		long long currentPos = 0;
		for (long long i = 0; i < positions.size() && currentPos < length; i++)
		{
			long long pos = positions[i];
			long long len = pos - currentPos;
			memmove(chars + currentPos + len + diff, chars + currentPos + len, length - len - currentPos + 1);
			memcpy(chars + currentPos + len, replacement, rLength);
			currentPos = pos + rLength;
			length += diff;
		}
	}

	void replaceAll(const char *c, const char *replacement)
	{
		std::vector<long long> positions;
		findAll(c, positions);
		int cLength = strlen(c);
		int rLength = strlen(replacement);
		int diff = rLength - cLength;
		long long currentPos = 0;
		for (long long i = 0; i < positions.size() && currentPos < length; i++)
		{
			long long pos = positions[i];
			long long len = pos - currentPos;
			memmove(chars + currentPos + len + diff, chars + currentPos + len, length - len - currentPos + 1);
			memcpy(chars + currentPos + len, replacement, rLength);
			currentPos = pos + rLength + diff;
			length += diff;
		}
	}
};


class myList
{
  private:
	myChar data;
	const char separator = '\n'; // 使用换行符作为元素分隔符

  public:
	myList() {}

	void append(const char *item)
	{
		data.cat(item);
		data.cat(&separator, 1);
	}

	const char *get(int index)
	{
		int current = 0;
		const char *start = data.chars;
		while (current < index)
		{
			start = strchr(start, separator);
			if (!start)
				return nullptr;
			start++;
			current++;
		}
		return start;
	}

	int length()
	{
		return std::count(data.chars, data.chars + data.length, separator);
	}

	void clear()
	{
		data.length = 0;
	}

	// 迭代器功能
	class Iterator
	{
	  private:
		const char *current;
		const char *end;

	  public:
		Iterator(const char *start, const char *end) : current(start), end(end) {}
		bool hasNext() { return current < end; }
		const char *next()
		{
			const char *result = current;
			current = strchr(current, '\n');
			if (current)
				current++;
			return result;
		}
	};

	Iterator iterator()
	{
		return Iterator(data.chars, data.chars + data.length);
	}
};

void charsView(struct myChar uc, int slices = 128, int width = 16)
{
	if (uc.length > 10000 * 100)
		printf("Size :%lf M,   %d sample slices with witdth=%d:\n", uc.length / 1000 / 1000.0, slices, width);
	else if (uc.length >= 10000)
		printf("Size :%lf K,   %d sample slices with witdth=%d:\n", uc.length / 1000.0, slices, width);
	else if (uc.length < 10000)
		printf("Size :%lld ,   %d sample slices with witdth=%d:\n", uc.length, slices, width);
	if (uc.length < slices * width)
		slices = 1;
	for (long long i = 0; i < uc.length; i = i + uc.length / slices)
	{
		for (long long j = 0; j < width; j++)
		{
			if (isprint(uc.chars[i + j]))
			{
				printf("%c", uc.chars[i + j]);
			}
			else
			{
				printf("[%X]", uc.chars[i + j]);
			}
		}
		printf("  -...- ");
	}
	printf("\n");
}
myChar readURL(char *url)
{
	myChar uc;
	unsigned long cmdLength = strlen("curl -s  \"\"") + strlen(url) + 1;
	char cmd[cmdLength];
	sprintf(cmd, "curl -s \"%s\"", url);

	FILE *furl = popen(cmd, "r");
	unsigned long total_size = 0;
	double t0, t;
	//uc.chars=NULL;
	//uc.length=0;
	//char temp_uc.chars[BUFFER_SIZE];
	char *temp_buffer = (char *)malloc(BUFFER_SIZE);

	while (1)
	{
		size_t bytes_read = fread(temp_buffer, 1, BUFFER_SIZE, furl);

		//printf("%lu , ",bytes_read);
		if (bytes_read <= 0 || temp_buffer == NULL)
			break;
		//printf(temp_buffer);
		unsigned long nReaded = bytes_read / sizeof(char);

		/*
uc.chars=(char*)realloc(uc.chars,uc.length+nReaded);

for (unsigned long i=uc.length;i<uc.length+nReaded;i++){
uc.chars[i]=temp_buffer[i-uc.length];
}

uc.length+=nReaded;	*/
		uc.cat(temp_buffer, bytes_read);
	}
	free(temp_buffer);
	//uc.chars[uc.length]='\0';
	return uc;
}

void writeToFile(char *path, char *data, long long len)
{
	FILE *output_file = fopen(path, "wb");
	if (output_file != NULL)
	{
		setbuf(output_file, NULL);
		fwrite(data, 1, len, output_file);
		fclose(output_file);
		printf("%lf MB Data written to %s\n", len / 1024 / 1024.0, path);
	}
	else
	{
		printf("Failed to open the output file.\n");
	}
}







struct myDict {
private:
    static const int INITIAL_SIZE = 4096;

    struct Entry {
        char* key;
        char* value;
        bool occupied;

        Entry() : key(nullptr), value(nullptr), occupied(false) {}
    };

    Entry* table;
    int table_size;
    int count;

    unsigned int hash(const char* key) const {
        unsigned int hash = 5381;
        int c;
        while ((c = *key++)) {
            hash = ((hash << 5) + hash) + c; // hash * 33 + c
        }
        return hash % table_size;
    }

    void resize() {
        int old_size = table_size;
        Entry* old_table = table;

        table_size *= 2;
        table = new Entry[table_size]();

        count = 0;
        for (int i = 0; i < old_size; i++) {
            if (old_table[i].occupied) {
                set(old_table[i].key, old_table[i].value);
                delete[] old_table[i].key;
                delete[] old_table[i].value;
            }
        }

        delete[] old_table;
    }

public:
    myDict() : table_size(INITIAL_SIZE), count(0) {
        table = new Entry[table_size]();
    }

    ~myDict() {
        clear();
        delete[] table;
    }

    myDict(const myDict& other) : table_size(other.table_size), count(other.count) {
        table = new Entry[table_size]();
        for (int i = 0; i < table_size; i++) {
            if (other.table[i].occupied) {
                table[i].key = strdup(other.table[i].key);
                table[i].value = strdup(other.table[i].value);
                table[i].occupied = true;
            }
        }
    }

    myDict& operator=(const myDict& other) {
        if (this != &other) {
            clear();
            delete[] table;

            table_size = other.table_size;
            count = other.count;
            table = new Entry[table_size]();
            for (int i = 0; i < table_size; i++) {
                if (other.table[i].occupied) {
                    table[i].key = strdup(other.table[i].key);
                    table[i].value = strdup(other.table[i].value);
                    table[i].occupied = true;
                }
            }
        }
        return *this;
    }

    void set(const char* key, const char* value) {
        if (count >= table_size * 0.75) {
            resize();
        }

        unsigned int index = hash(key);

        for (int i = 0; i < table_size; i++) {
            unsigned int probeIndex = (index + i) % table_size;
            if (!table[probeIndex].occupied || (table[probeIndex].key && strcmp(table[probeIndex].key, key) == 0)) {
                if (table[probeIndex].key) {
                    delete[] table[probeIndex].key;
                    delete[] table[probeIndex].value;
                }

                table[probeIndex].key = strdup(key);
                table[probeIndex].value = strdup(value);

                if (!table[probeIndex].occupied) {
                    count++;
                }
                table[probeIndex].occupied = true;
                return;
            }
        }
    }

    const char* get(const char* key) const {
        unsigned int index = hash(key);

        for (int i = 0; i < table_size; i++) {
            unsigned int probeIndex = (index + i) % table_size;
            if (!table[probeIndex].occupied) {
                return nullptr;
            }
            if (strcmp(table[probeIndex].key, key) == 0) {
                return table[probeIndex].value;
            }
        }
        return nullptr;
    }

    void delete_key(const char* key) {
        unsigned int index = hash(key);

        for (int i = 0; i < table_size; i++) {
            unsigned int probeIndex = (index + i) % table_size;
            if (!table[probeIndex].occupied) {
                return;  // Key not found
            }
            if (strcmp(table[probeIndex].key, key) == 0) {
                delete[] table[probeIndex].key;
                delete[] table[probeIndex].value;
                table[probeIndex].occupied = false;
                count--;
                return;
            }
        }
    }

    void clear() {
        for (int i = 0; i < table_size; i++) {
            if (table[i].occupied) {
                delete[] table[i].key;
                delete[] table[i].value;
                table[i].occupied = false;
            }
        }
        count = 0;
    }

    int getCount() const { return count; }

    void debug_print() const {
        for (int i = 0; i < table_size; i++) {
            if (table[i].occupied) {
                printf("%s: %s\n", table[i].key, table[i].value);
            }
        }
    }

    // Iterator classes (KeyIterator, ValueIterator, and ItemIterator) remain the same
    // ...

    class ItemIterator {
    private:
        const myDict& dict;
        int current_index;

        void findNextOccupied() {
            while (current_index < dict.table_size && !dict.table[current_index].occupied) {
                ++current_index;
            }
        }

    public:
        ItemIterator(const myDict& d, int start_index = 0) : dict(d), current_index(start_index) {
            findNextOccupied();
        }

        std::pair<const char*, const char*> operator*() const {
            if (current_index >= dict.table_size) {
                throw std::out_of_range("Iterator out of range");
            }
            return {dict.table[current_index].key, dict.table[current_index].value};
        }

        ItemIterator& operator++() {
            if (current_index < dict.table_size) {
                ++current_index;
                findNextOccupied();
            }
            return *this;
        }

        bool operator!=(const ItemIterator& other) const {
            return current_index != other.current_index;
        }
    };

    ItemIterator begin_items() const { return ItemIterator(*this); }
    ItemIterator end_items() const { return ItemIterator(*this, table_size); }
};

#endif // MYDATA_H