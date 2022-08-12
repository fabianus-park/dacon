# pip install dacon_submit_api-0.0.4-py3-none-any.whl
from dacon_submit_api import dacon_submit_api 
import local-setting

result = dacon_submit_api.post_submission_file(
'파일경로', 
dacon_token, 
'대회ID', 
'팀이름', 
'submission 메모 내용' );
