#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <algorithm>
#include <functional>
#include <string_view>

using namespace std;

/**
 * Camel for Class
 * snake for the rest
*/


/**
 * struct Comparator{
 * // operator overloading
 * bool operator() (int a, int b) const{
 * // sort in ascending orderretunr a < b;}
 */
struct Comparator{
    bool operator() (int a, int b) const{
        // sort in ascending order
        return a < b;
    }
};

class Sorting{

    public:
    // constrcutor
    Sorting(vector<int>& nums): nums(nums){};

    // default destructor
    ~Sorting() = default;

    std::function<void(string_view const)> print = [this](string_view const message){
        for (auto element: this->nums){
            cout << element << endl;
        }
        cout << ": " << message << endl;
    };

    int partition(vector<int>& nums, int low, int high){
        // select pivot element
        int pivot = nums[high];
        // index boundary to seperate cluster smaller than pivot and greater than
        int i = low - 1; // -1 prior to swaping
        // categorize
        for (int j = low; j < high; j++){
            if (nums[j] < pivot){
                i++;
                // swap the skipped greater elements with the current found smaller element position
                swap(nums[i], nums[j]);
            }
        }
        return i + 1;
    };
    /** methods with lambda functions */
    void quickSort(vector<int>& nums, int low_id, int high_id){
        /** time complexity: o(nlogn) */
        // two cluster pivoting
        if(low_id < high_id){
            int pivot = partition(nums, low_id, high_id);
            // sort the lower part
            quickSort(nums, low_id, pivot - 1);
            // sort the higher part
            quickSort(nums, pivot + 1, high_id);
        }
    };

    void mergeSort(vector <int>& nums, int low_id, int high_id){
        /** time complexity: o(nlogn) */
        // divide and conquer
        // categoraize into two vectors with same lengt
        if (low_id < high_id){ 
            int mid_id = low_id + static_cast<int>((high_id - low_id) / 2);
            // regressively deompose until the subset length is 1
            mergeSort(nums, low_id, mid_id);
            mergeSort(nums,mid_id + 1, high_id);
            // initialize two new vectors left and right for later merging
            int n1 = mid_id - low_id + 1;
            int n2 = high_id - mid_id;
            vector<int> L(n1);
            vector<int> R(n2);
            std::copy(nums.begin() + low_id, nums.begin() + mid_id,  L.begin());
            std::copy(nums.begin() + mid_id + 1, nums.begin() + high_id + 1, R.begin());

            // compare and merge the two vectors
            int i = 0;
            int j = 0;
            int k = low_id;

            while(i < n1 && j < n2){
                if (L[i] <= R[j]){
                    // assign the current smaller cell value to the original vector
                    nums[k] = L[i];
                    i++;
                }else{
                    nums[k] = R[j];
                    j++;
                }
                k++;
            }
            while (i < n1){
                nums[k] = L[i];
                i++;
                k++;
            }
            while(j < n2){
                nums[k] = R[j];
                j++;
                k++;
            }
        }

    }

    void heapify(vector<int>& nums, int n, int i){
        int largest_id = i;
        int left_id = 2 * i + 1;
        int right_id = 2 * i + 2;
        // find the largest element index from the left if in bound
        if ( left_id < n && nums[left_id] > nums[largest_id]){
            largest_id = left_id;
        }
        // find the largest element index from the right if in bound
        if (right_id < n &&  nums[right_id] > nums[largest_id]){
            largest_id = right_id;
        }
        if (largest_id != i){
            swap(nums[i], nums[largest_id]);
            heapify(nums, n, largest_id);
        }
    }

    void heapSort(vector<int>& nums){
        /** time complexity: o(nlogn) */
        // initialize heap memory
        int n = nums.size();
        // heapify from root node (left leafs, root node, right leafs)
        for (int i = (n / 2) - 1; i >= 0; i--){
            // each time heapify will look at the next two elements and keep on swapping the node values larger than current value to the current position
            heapify(nums, n, i);
        }
        // swap the maximum element with last element
        for (int i = n - 1 ;  i > 0; i --){
            swap(nums[0], nums[i]);
            heapify(nums, i, 0);
        }
    }

    vector<int>& getNums(){
        return nums;
    }

    private:
    vector<int>& nums;

};

class Search{
    public:
    Search(){};

    template<typename T>
    int binary_search(const vector<T>& arr, const T& target){
        /** time complexity: o(nlogn) */
        int lowest_idx = 0;
        int highest_idx = arr.size() - 1;
        // left and right search
        while (lowest_idx <= highest_idx){
            int mid_idx = (lowest_idx + highest_idx) / 2;
            if (arr[mid_idx] == target){
                return mid_idx;
            }
            else if (arr[mid_idx] < target){
                lowest_idx = mid_idx + 1;
            }
            else {
                highest_idx = mid_idx - 1;
            }
        }
        return -1;
    }

    private:

};

int main(){
    std::vector<int> nums = {5, 2, 9, 1, 5, 6};
    Sorting sorting(nums);

    sorting.heapSort(sorting.getNums());
    sorting.print("Hello, World!");

    for_each(nums.begin(), nums.end(), [](int num){
        cout << num << endl;
    });

    Search searching;
    sort(nums.begin(), nums.end(), [](int a, int b){return a > b;});
    int target = 5;
    int result = searching.binary_search(nums, target);
    cout << result << endl;


}