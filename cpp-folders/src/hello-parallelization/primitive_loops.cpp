#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

int main()
{
    vector<int> array(100);
    for (int i = 0; i < array.size(); i++)
    {
        array[i] = rand() % 100;
    }

    for (int i = 0; i < array.size(); i++)
    {
        cout << array[i] << " ";
    }
    cout << endl;


    auto task = [](int &element)
    {
        element += 1;
        element *= 2;
    };

    for_each(array.begin(), array.end(), task);


    for (int i = 0; i < array.size(); i++)
    {
        cout << array[i] << " ";
    }
    cout << endl;

    return 0;
}