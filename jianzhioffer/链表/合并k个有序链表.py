# coding=ut-8
import heapq
# 堆，找到k个节点中的最小值
# 时复nlogk    n是k个链表的节点数和
class Solution(object):
    def mergeKLists(self, lists):
        head = ListNode(-1)
        current = head
        heap = []
        for node in lists:
            if node:
                heap.append((node.val, node))
        heapq.heapify(heap)  # 堆维护，时复klogk
        while heap:
            _, ref = heapq.heappop(heap)  # value and node
            current.next = ref
            current = current.next
            if ref.next:  # 入堆的那个链表的下一个节点入heap
                heapq.heappush(heap, (ref.next.val, ref.next))
        return head.next