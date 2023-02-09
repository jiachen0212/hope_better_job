// # 字符串生成二叉树
// https://www.cnblogs.com/grandyang/p/6793904.html

/*
当前位置开始遍历，因为当前位置是一个左括号，我们的目标是找到与之对应的右括号的位置，
但是由于中间还会遇到左右括号，所以我们需要用一个变量 cnt 来记录左括号的个数，
如果遇到左括号，cnt 自增1，如果遇到右括号，cnt 自减1，这样当某个时刻 cnt 为0的时候，
我们就确定了一个完整的子树的位置，
那么问题来了，这个子树到底是左子树还是右子树呢，
我们需要一个辅助变量 start，当最开始找到第一个左括号的位置时，将 start 赋值为该位置，
那么当 cnt 为0时，如果 start 还是原来的位置，说明这个是左子树，
我们对其调用递归函数，注意此时更新 start 的位置，这样就能区分左右子树了
*/

class Solution {
public:
    TreeNode* str2tree(string s) {
        if (s.empty())
            return NULL;
        auto found = s.find('(');
        // found == string::npos 没找到?
        int val = (found == string::npos) ? stoi(s) : stoi(s.substr(0, found));
        TreeNode *cur = new TreeNode(val);
        if (found == string::npos)
            return cur;
        int start = found, cnt = 0;
        for (int i = start; i < s.size(); ++i) {
            if (s[i] == '(') ++cnt;
            else if (s[i] == ')') --cnt;
            if (cnt == 0 && start == found) {
                cur->left = str2tree(s.substr(start + 1, i - start - 1));
                start = i + 1;
            } else if (cnt == 0) {
                cur->right = str2tree(s.substr(start + 1, i - start - 1));
            }
        }
        return cur;
    }
};




