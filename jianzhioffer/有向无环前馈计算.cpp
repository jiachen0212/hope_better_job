struct My_ListNode{
    int idx;            //表示图节点的编号
    int val;            //表示图节点上的值
    bool visited;       //判断当前节点是否已经遍历过
    My_ListNode* next;  //指向下一个节点

};

//加减乘除四项操作，返回到value
int operate(int &value1, int value2, char operator_math){
    //TODO
}
/*
    图定义：  用邻接表 + 逆邻接表表示，一个节点既有入度的链表，又有出度的链表。
    顶点的定义 My_ListNode： [index, val, false, next].
*/
int ByteDance_Graph(vector<My_ListNode*> vertex_in, vector<My_ListNode*> vertex_out, vector<char> operator_math){
    //对入度vector进行遍历
    for(int i=0; i<vertex_in.size(); i++){
        ListNode* in_cur = vertex_in[i];
        //每个节点的入度链表进行遍历
        while(in_cur){
            int in_cur_index = in_cur->idx;
            int in_cur_val = in_cur->val;
            bool in_cur_visited = in_cur->visited;

            //如果当前节点已经访问过，则不继续用它的值去更新它的入度节点
            if(in_cur_visited){
                //用编号找到当前节点的出度节点
                out_cur_node = vertex_out[in_cur_index];
                while(out_cur_node){
                    //更新节点访问标签
                    out_cur_node->visited = True;
                    //更新节点的值
                    out_cur_node->val = operate(out_cur_node->val, in_cur_val, operator_math[in_cur_index]);
                    //遍历当前节点的下一个出度节点
                    out_cur_node = out_cur_node->next;
                }
                in_cur = in_cur->next;
            }
        }
    }
}