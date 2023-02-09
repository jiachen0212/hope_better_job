# coding=utf-8
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        '''
        # 求「在一个数组的某个元素的右边，比自己小的元素的个数」
        # 归并排序过程中，已经出去的数，都是比之后小的数，记录他们的index
        '''
        if not nums:
            return []
        ll = len(nums)
        res = [0]*ll
        if ll == 1:
            return res
        temp = [-1 for _ in range(ll)] # 存放归并处理过程中的数组，一个tmp量
        # 索引数组，作用：归并回去的时候，方便知道是哪个下标的元素
        indexes = [i for i in range(ll)]

        self.merge_and_count_smaller(nums, 0, ll-1, temp, indexes, res)
        return res

    def merge_and_count_smaller(self, nums, left, right, temp, indexes, res):
        if left == right:
            return
        mid = (right+left) // 2
        self.merge_and_count_smaller(nums, left, mid, temp, indexes, res)
        self.merge_and_count_smaller(nums, mid+1, right, temp, indexes, res)

        # 代码走到这里的时候，[left, mid] 和 [mid + 1, right] 已经完成了排序并且计算好右侧小于每个元素的个数
        if nums[indexes[mid]] <= nums[indexes[mid + 1]]:
            # 此时不用计算横跨两个区间的右侧小于每个元素的个数，直接返回
            # 为什么不用？当前if条件满足时，说明[mid + 1, right]所有数字都比[left, mid]的大，继续计算右侧小于每个元素的个数没有意义，相当于剪枝。
            return
        self.sort_and_count_smaller(nums, left, mid, right, temp, indexes, res)

    def sort_and_count_smaller(self, nums, left, mid, right, temp, indexes, res):
        # [left,mid] 前有序数组, [mid+1,right] 后有序数组
        # 先拷贝，再合并
        # 由于前数组和后数组都有序，此时若出现后数组元素较大或者指针出界的情况，说明之前存在后数组元素较小的情况。需要计算后数组小于当前元素的个数并累计到res中。
        for i in range(left, right + 1):
            temp[i] = indexes[i]
        i = left
        j = mid + 1
        for k in range(left, right + 1):
            # 每一次合并nums，都需要判断数组下标是否越界。
            if i > mid:
                # i > mid 表示 i 已经遍历完了第一个部分的所有数，mid 是第一个部分的最后一个位置的下标。所以当前合并过程可以忽略i坐标，直接把j坐标所在元素合并到 indexes。
                indexes[k] = temp[j]
                j += 1
            elif j > right:
                # 表示 j 已经遍历完了第二个部分的所有数， right 是第二个部分的最后一个位置的下标。所以当前合并过程可以忽略j坐标，直接把i坐标所在元素合并到 indexes。
                indexes[k] = temp[i]
                i += 1
                # 后数组中的所有数字都比k对应的元素小；把结果放入res
                res[indexes[k]] += (right - mid)
            # 走到这里时，说明i，j都没有越界，可以直接比较、合并到 indexes。
            elif nums[temp[i]] <= nums[temp[j]]:
                # 此时前数组元素出列，需要统计右侧小于当前元素的个数
                indexes[k] = temp[i]
                i += 1
                # 后数组中的[mid, j - 1]对应都比k对应的元素小（很显然不包含当前j元素）；把结果放入res
                res[indexes[k]] += (j - mid - 1)
            else:
                # 此时后数组元素出列，不统计右侧小于当前元素的个数
                indexes[k] = temp[j]
                j += 1