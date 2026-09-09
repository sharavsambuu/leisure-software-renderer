#include <iostream>
#include <vector>
#include <thread>

using namespace std;


int main()
{
    vector<int> array(100);
    for (int i = 0; i < array.size(); i++)
    {
        array[i] = rand() % 100;
    }

    auto task = [](int &element)
    {
        element += 1;
        element *= 2;
    };

    vector<thread> threads;
    for (int i = 0; i < array.size(); i++)
    {
        threads.push_back(thread(task, ref(array[i])));
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    for (int i = 0; i < array.size(); i++)
    {
        cout << array[i] << " ";
    }
    cout << endl;

    return 0;
}