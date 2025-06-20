import copy
import math
import os

from .utils import md5
from .input_event import TouchEvent, LongTouchEvent, ScrollEvent, SetTextEvent, KeyEvent

import requests
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

solo_text = False
choice_LLM = False
last_actions = []

#droidbot -a C:\Users\juve2\StudioProjects\PassAndroid\android\build\intermediates\apk\noMapsNoAnalyticsForFDroid\debug\PassAndroid-3.7.3-noMaps-noAnalytics-forFDroid-debug.apk -o output-pass-llama -is_emulator -accessibility_auto -timeout 10800
#droidbot -a C:\Users\juve2\StudioProjects\Omni-Notes\omniNotes\build\intermediates\apk\alpha\debug\OmniNotes-alphaDebug-6.4.0.apk  -o output-note-llama -is_emulator -accessibility_auto -timeout 10800
#droidbot -a C:\Users\juve2\StudioProjects\thunderbird-android\app-k9mail\build\outputs\apk\foss\debug\app-k9mail-foss-debug.apk  -o output-tfa-llama -is_emulator -accessibility_auto -timeout 10800

# Load the tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

model_id = "meta-llama/Llama-3.2-1B-Instruct"
#model_id = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)




#we define a method to ask any prompt to llama
def ask_llama_local(prompt, maxl=600, temp=0.7):
    """
    Send a prompt to the Llama model and get a response.

    Args:
    - prompt (str): The input question or statement to the model.
    - max_length (int): The maximum length of the response.
    - temperature (float): Controls randomness in the model's output.

    Returns:
    - str: The model's generated response.
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    print("Lughezza input" ,len(inputs['input_ids'][0]))
    maxl = len(inputs['input_ids'][0]) + 5
    print("maxl: ", maxl)

    inputs.to(device)

    # Generate the output
    outputs = model.generate(
        inputs['input_ids'],  # Tokenized input
        temperature=temp,        # Lower temperature to reduce randomness
        do_sample=True,        # Disable sampling for deterministic output
        pad_token_id=tokenizer.eos_token_id  # Ensure the model doesn't go beyond the end token
    )

    # Decode and return the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

print("Llama model loaded successfully!")
print("Test the model with a sample prompt:")
prompt = "What is the capital of France?"
response = ask_llama_local(prompt)
print("Response from Llama model:", response)



class DeviceState(object):
    """
    the state of the current device
    """

    def __init__(self, device, views, foreground_activity, activity_stack, background_services,
                 tag=None, screenshot_path=None):
        self.device = device
        self.foreground_activity = foreground_activity
        self.activity_stack = activity_stack if isinstance(activity_stack, list) else []
        self.background_services = background_services
        if tag is None:
            from datetime import datetime
            tag = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.tag = tag
        self.screenshot_path = screenshot_path
        self.views = self.__parse_views(views)
        self.view_tree = {}
        self.__assemble_view_tree(self.view_tree, self.views)
        self.__generate_view_strs()
        self.state_str = self.__get_state_str()
        self.structure_str = self.__get_content_free_state_str()
        self.search_content = self.__get_search_content()
        self.text_representation = self.get_text_representation()
        self.possible_events = None
        self.width = device.get_width(refresh=True)
        self.height = device.get_height(refresh=False)

    @property
    def activity_short_name(self):
        return self.foreground_activity.split('.')[-1]

    def to_dict(self):
        state = {'tag': self.tag,
                 'state_str': self.state_str,
                 'state_str_content_free': self.structure_str,
                 'foreground_activity': self.foreground_activity,
                 'activity_stack': self.activity_stack,
                 'background_services': self.background_services,
                 'width': self.width,
                 'height': self.height,
                 'views': self.views}
        return state

    def to_json(self):
        import json
        return json.dumps(self.to_dict(), indent=2)

    def __parse_views(self, raw_views):
        views = []
        if not raw_views or len(raw_views) == 0:
            return views

        for view_dict in raw_views:
            # # Simplify resource_id
            # resource_id = view_dict['resource_id']
            # if resource_id is not None and ":" in resource_id:
            #     resource_id = resource_id[(resource_id.find(":") + 1):]
            #     view_dict['resource_id'] = resource_id
            views.append(view_dict)
        return views

    def __assemble_view_tree(self, root_view, views):
        if not len(self.view_tree): # bootstrap
            if not len(views): # to fix if views is empty
                return
            self.view_tree = copy.deepcopy(views[0])
            self.__assemble_view_tree(self.view_tree, views)
        else:
            children = list(enumerate(root_view["children"]))
            if not len(children):
                return
            for i, j in children:
                root_view["children"][i] = copy.deepcopy(self.views[j])
                self.__assemble_view_tree(root_view["children"][i], views)

    def __generate_view_strs(self):
        for view_dict in self.views:
            self.__get_view_str(view_dict)
            # self.__get_view_structure(view_dict)

    @staticmethod
    def __calculate_depth(views):
        root_view = None
        for view in views:
            if DeviceState.__safe_dict_get(view, 'parent') == -1:
                root_view = view
                break
        DeviceState.__assign_depth(views, root_view, 0)

    @staticmethod
    def __assign_depth(views, view_dict, depth):
        view_dict['depth'] = depth
        for view_id in DeviceState.__safe_dict_get(view_dict, 'children', []):
            DeviceState.__assign_depth(views, views[view_id], depth + 1)

    def __get_state_str(self):
        state_str_raw = self.__get_state_str_raw()
        return md5(state_str_raw)

    def __get_state_str_raw(self):
        if self.device.humanoid is not None:
            import json
            from xmlrpc.client import ServerProxy
            proxy = ServerProxy("http://%s/" % self.device.humanoid)
            return proxy.render_view_tree(json.dumps({
                "view_tree": self.view_tree,
                "screen_res": [self.device.display_info["width"],
                               self.device.display_info["height"]]
            }))
        else:
            view_signatures = set()
            for view in self.views:
                view_signature = DeviceState.__get_view_signature(view)
                if view_signature:
                    view_signatures.add(view_signature)
            return "%s{%s}" % (self.foreground_activity, ",".join(sorted(view_signatures)))

    def __get_content_free_state_str(self):
        if self.device.humanoid is not None:
            import json
            from xmlrpc.client import ServerProxy
            proxy = ServerProxy("http://%s/" % self.device.humanoid)
            state_str = proxy.render_content_free_view_tree(json.dumps({
                "view_tree": self.view_tree,
                "screen_res": [self.device.display_info["width"],
                               self.device.display_info["height"]]
            }))
        else:
            view_signatures = set()
            for view in self.views:
                view_signature = DeviceState.__get_content_free_view_signature(view)
                if view_signature:
                    view_signatures.add(view_signature)
            state_str = "%s{%s}" % (self.foreground_activity, ",".join(sorted(view_signatures)))
        import hashlib
        return hashlib.md5(state_str.encode('utf-8')).hexdigest()

    def __get_search_content(self):
        """
        get a text for searching the state
        :return: str
        """
        words = [",".join(self.__get_property_from_all_views("resource_id")),
                 ",".join(self.__get_property_from_all_views("text"))]
        return "\n".join(words)

    def __get_property_from_all_views(self, property_name):
        """
        get the values of a property from all views
        :return: a list of property values
        """
        property_values = set()
        for view in self.views:
            property_value = DeviceState.__safe_dict_get(view, property_name, None)
            if property_value:
                property_values.add(property_value)
        return property_values

    def save2dir(self, output_dir=None):
        try:
            if output_dir is None:
                if self.device.output_dir is None:
                    return
                else:
                    output_dir = os.path.join(self.device.output_dir, "states")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            dest_state_json_path = "%s/state_%s.json" % (output_dir, self.tag)
            if self.device.adapters[self.device.minicap]:
                dest_screenshot_path = "%s/screen_%s.jpg" % (output_dir, self.tag)
            else:
                dest_screenshot_path = "%s/screen_%s.png" % (output_dir, self.tag)
            state_json_file = open(dest_state_json_path, "w")
            state_json_file.write(self.to_json())
            state_json_file.close()
            import shutil
            shutil.copyfile(self.screenshot_path, dest_screenshot_path)
            self.screenshot_path = dest_screenshot_path
            # from PIL.Image import Image
            # if isinstance(self.screenshot_path, Image):
            #     self.screenshot_path.save(dest_screenshot_path)
        except Exception as e:
            self.device.logger.warning(e)

    def save_view_img(self, view_dict, output_dir=None):
        try:
            if output_dir is None:
                if self.device.output_dir is None:
                    return
                else:
                    output_dir = os.path.join(self.device.output_dir, "views")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            view_str = view_dict['view_str']
            if self.device.adapters[self.device.minicap]:
                view_file_path = "%s/view_%s.jpg" % (output_dir, view_str)
            else:
                view_file_path = "%s/view_%s.png" % (output_dir, view_str)
            if os.path.exists(view_file_path):
                return
            from PIL import Image
            # Load the original image:
            view_bound = view_dict['bounds']
            original_img = Image.open(self.screenshot_path)
            # view bound should be in original image bound
            view_img = original_img.crop((min(original_img.width - 1, max(0, view_bound[0][0])),
                                          min(original_img.height - 1, max(0, view_bound[0][1])),
                                          min(original_img.width, max(0, view_bound[1][0])),
                                          min(original_img.height, max(0, view_bound[1][1]))))
            view_img.convert("RGB").save(view_file_path)
        except Exception as e:
            self.device.logger.warning(e)

    def is_different_from(self, another_state):
        """
        compare this state with another
        @param another_state: DeviceState
        @return: boolean, true if this state is different from other_state
        """
        return self.state_str != another_state.state_str

    @staticmethod
    def __get_view_signature(view_dict):
        """
        get the signature of the given view
        @param view_dict: dict, an element of list DeviceState.views
        @return:
        """
        if 'signature' in view_dict:
            return view_dict['signature']

        view_text = DeviceState.__safe_dict_get(view_dict, 'text', "None")
        if view_text is None or len(view_text) > 50:
            view_text = "None"

        signature = "[class]%s[resource_id]%s[text]%s[%s,%s,%s]" % \
                    (DeviceState.__safe_dict_get(view_dict, 'class', "None"),
                     DeviceState.__safe_dict_get(view_dict, 'resource_id', "None"),
                     view_text,
                     DeviceState.__key_if_true(view_dict, 'enabled'),
                     DeviceState.__key_if_true(view_dict, 'checked'),
                     DeviceState.__key_if_true(view_dict, 'selected'))
        view_dict['signature'] = signature
        return signature

    @staticmethod
    def __get_content_free_view_signature(view_dict):
        """
        get the content-free signature of the given view
        @param view_dict: dict, an element of list DeviceState.views
        @return:
        """
        if 'content_free_signature' in view_dict:
            return view_dict['content_free_signature']
        content_free_signature = "[class]%s[resource_id]%s" % \
                                 (DeviceState.__safe_dict_get(view_dict, 'class', "None"),
                                  DeviceState.__safe_dict_get(view_dict, 'resource_id', "None"))
        view_dict['content_free_signature'] = content_free_signature
        return content_free_signature

    def __get_view_str(self, view_dict):
        """
        get a string which can represent the given view
        @param view_dict: dict, an element of list DeviceState.views
        @return:
        """
        if 'view_str' in view_dict:
            return view_dict['view_str']
        view_signature = DeviceState.__get_view_signature(view_dict)
        parent_strs = []
        for parent_id in self.get_all_ancestors(view_dict):
            parent_strs.append(DeviceState.__get_view_signature(self.views[parent_id]))
        parent_strs.reverse()
        child_strs = []
        for child_id in self.get_all_children(view_dict):
            child_strs.append(DeviceState.__get_view_signature(self.views[child_id]))
        child_strs.sort()
        view_str = "Activity:%s\nSelf:%s\nParents:%s\nChildren:%s" % \
                   (self.foreground_activity, view_signature, "//".join(parent_strs), "||".join(child_strs))
        import hashlib
        view_str = hashlib.md5(view_str.encode('utf-8')).hexdigest()
        view_dict['view_str'] = view_str
        return view_str

    def __get_view_structure(self, view_dict):
        """
        get the structure of the given view
        :param view_dict: dict, an element of list DeviceState.views
        :return: dict, representing the view structure
        """
        if 'view_structure' in view_dict:
            return view_dict['view_structure']
        width = DeviceState.get_view_width(view_dict)
        height = DeviceState.get_view_height(view_dict)
        class_name = DeviceState.__safe_dict_get(view_dict, 'class', "None")
        children = {}

        root_x = view_dict['bounds'][0][0]
        root_y = view_dict['bounds'][0][1]

        child_view_ids = self.__safe_dict_get(view_dict, 'children')
        if child_view_ids:
            for child_view_id in child_view_ids:
                child_view = self.views[child_view_id]
                child_x = child_view['bounds'][0][0]
                child_y = child_view['bounds'][0][1]
                relative_x, relative_y = child_x - root_x, child_y - root_y
                children["(%d,%d)" % (relative_x, relative_y)] = self.__get_view_structure(child_view)

        view_structure = {
            "%s(%d*%d)" % (class_name, width, height): children
        }
        view_dict['view_structure'] = view_structure
        return view_structure

    @staticmethod
    def __key_if_true(view_dict, key):
        return key if (key in view_dict and view_dict[key]) else ""

    @staticmethod
    def __safe_dict_get(view_dict, key, default=None):
        value = view_dict[key] if key in view_dict else None
        return value if value is not None else default

    @staticmethod
    def get_view_center(view_dict):
        """
        return the center point in a view
        @param view_dict: dict, an element of DeviceState.views
        @return: a pair of int
        """
        bounds = view_dict['bounds']
        return (bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2

    @staticmethod
    def get_view_width(view_dict):
        """
        return the width of a view
        @param view_dict: dict, an element of DeviceState.views
        @return: int
        """
        bounds = view_dict['bounds']
        return int(math.fabs(bounds[0][0] - bounds[1][0]))

    @staticmethod
    def get_view_height(view_dict):
        """
        return the height of a view
        @param view_dict: dict, an element of DeviceState.views
        @return: int
        """
        bounds = view_dict['bounds']
        return int(math.fabs(bounds[0][1] - bounds[1][1]))

    def get_all_ancestors(self, view_dict):
        """
        Get temp view ids of the given view's ancestors
        :param view_dict: dict, an element of DeviceState.views
        :return: list of int, each int is an ancestor node id
        """
        result = []
        parent_id = self.__safe_dict_get(view_dict, 'parent', -1)
        if 0 <= parent_id < len(self.views):
            result.append(parent_id)
            result += self.get_all_ancestors(self.views[parent_id])
        return result

    def get_all_children(self, view_dict):
        """
        Get temp view ids of the given view's children
        :param view_dict: dict, an element of DeviceState.views
        :return: set of int, each int is a child node id
        """
        children = self.__safe_dict_get(view_dict, 'children')
        if not children:
            return set()
        children = set(children)
        for child in children:
            children_of_child = self.get_all_children(self.views[child])
            children.union(children_of_child)
        return children

    def get_app_activity_depth(self, app):
        """
        Get the depth of the app's activity in the activity stack
        :param app: App
        :return: the depth of app's activity, -1 for not found
        """
        depth = 0
        for activity_str in self.activity_stack:
            if app.package_name in activity_str:
                return depth
            depth += 1
        return -1

    def get_possible_input(self):
        """
        Get a list of possible input events for this state
        :return: list of InputEvent
        """
        text = False
        possible_events = []
        enabled_view_ids = []
        touch_exclude_view_ids = set()
        for view_dict in self.views:
            # exclude navigation bar if exists
            if self.__safe_dict_get(view_dict, 'enabled') and \
                    self.__safe_dict_get(view_dict, 'visible') and \
                    self.__safe_dict_get(view_dict, 'resource_id') not in \
               ['android:id/navigationBarBackground',
                'android:id/statusBarBackground']:
                enabled_view_ids.append(view_dict['temp_id'])
        # enabled_view_ids.reverse()

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'clickable'):
                possible_events.append(TouchEvent(view=self.views[view_id]))
                touch_exclude_view_ids.add(view_id)
                touch_exclude_view_ids.union(self.get_all_children(self.views[view_id]))

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'scrollable'):
                possible_events.append(ScrollEvent(view=self.views[view_id], direction="up"))
                possible_events.append(ScrollEvent(view=self.views[view_id], direction="down"))
                possible_events.append(ScrollEvent(view=self.views[view_id], direction="left"))
                possible_events.append(ScrollEvent(view=self.views[view_id], direction="right"))

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'checkable'):
                possible_events.append(TouchEvent(view=self.views[view_id]))
                touch_exclude_view_ids.add(view_id)
                touch_exclude_view_ids.union(self.get_all_children(self.views[view_id]))

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'long_clickable'):
                possible_events.append(LongTouchEvent(view=self.views[view_id]))

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'editable'):
                if solo_text:
                    text = True
                full_view_representation = self.text_representation[0]
                text_to_set = self.ask_llama(view=self.views[view_id], full_view_representation=full_view_representation)
                print("text_to_set: ", text_to_set)
                event = SetTextEvent(view=self.views[view_id], text=text_to_set)
                #text_to_set = "HelloWorld"  # TODO: replace with actual text generation
                possible_events.append(SetTextEvent(view=self.views[view_id], text=text_to_set))
                touch_exclude_view_ids.add(view_id)
                # TODO figure out what event can be sent to editable views
                pass

        for view_id in enabled_view_ids:
            if view_id in touch_exclude_view_ids:
                continue
            children = self.__safe_dict_get(self.views[view_id], 'children')
            if children and len(children) > 0:
                continue
            possible_events.append(TouchEvent(view=self.views[view_id]))

        # For old Android navigation bars
        # possible_events.append(KeyEvent(name="MENU"))
        
        if text:
            possible_events = [event]
            print("Text input event is available")
            print("Possible events: ", possible_events)
        if choice_LLM:
            possible_events = self.ask_choice_llama(view=self.views[0], full_view_representation=self.text_representation[0], possible_actions=possible_events)
        self.possible_events = possible_events
        return [] + possible_events
    
    def _get_ancestor_id(self, view, key):
        """
        Ritorna l'id dell'antenato più vicino che soddisfa view[key]==True,
        o -1 se non ne esistono.
        """
        curr_id = view['temp_id']
        # scorre verso l'alto finché non trova il flag
        while curr_id != -1:
            v = self.views[curr_id]
            if v.get(key):
                return curr_id
            curr_id = v.get('parent', -1)
        return -1

    def _extract_all_children(self, id):
        """
        Estrae ricorsivamente tutti gli id dei figli di una view.
        """
        result = set()
        to_visit = [id]
        while to_visit:
            vid = to_visit.pop()
            children = self.views[vid].get('children', [])
            for c in children:
                if c not in result:
                    result.add(c)
                    to_visit.append(c)
        return result
    
    def get_text_representation(self, merge_buttons=False):
        """
        Get a text representation of current state
        """
        enabled_view_ids = []
        for view_dict in self.views:
            # exclude navigation bar if exists
            if self.__safe_dict_get(view_dict, 'visible') and \
                self.__safe_dict_get(view_dict, 'resource_id') not in \
               ['android:id/navigationBarBackground',
                'android:id/statusBarBackground']:
                enabled_view_ids.append(view_dict['temp_id'])
        
        text_frame = "<p id=@ text='&' attr=null bounds=null>#</p>"
        btn_frame = "<button id=@ text='&' attr=null bounds=null>#</button>"
        checkbox_frame = "<checkbox id=@ text='&' attr=null bounds=null>#</checkbox>"
        input_frame = "<input id=@ text='&' attr=null bounds=null>#</input>"
        scroll_frame = "<scrollbar id=@ attr=null bounds=null></scrollbar>"
        

        view_descs = []
        indexed_views = []
        # available_actions = []
        removed_view_ids = []

        for view_id in enabled_view_ids:
            if view_id in removed_view_ids:
                continue
            # print(view_id)
            view = self.views[view_id]
            clickable = self._get_self_ancestors_property(view, 'clickable')
            scrollable = self.__safe_dict_get(view, 'scrollable')
            checkable = self._get_self_ancestors_property(view, 'checkable')
            long_clickable = self._get_self_ancestors_property(view, 'long_clickable')
            editable = self.__safe_dict_get(view, 'editable')
            actionable = clickable or scrollable or checkable or long_clickable or editable
            checked = self.__safe_dict_get(view, 'checked', default=False)
            selected = self.__safe_dict_get(view, 'selected', default=False)
            content_description = self.__safe_dict_get(view, 'content_description', default='')
            view_text = self.__safe_dict_get(view, 'text', default='')
            view_class = self.__safe_dict_get(view, 'class').split('.')[-1]
            bounds = self.__safe_dict_get(view, 'bounds')
            view_bounds = f'{bounds[0][0]},{bounds[0][1]},{bounds[1][0]},{bounds[1][1]}'
            if not content_description and not view_text and not scrollable:  # actionable?
                continue

            # text = self._merge_text(view_text, content_description)
            # view_status = ''
            view_local_id = str(len(view_descs))
            if editable:
                view_desc = input_frame.replace('@', view_local_id).replace('#', view_text)
                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" text='&'", "")
                # available_actions.append(SetTextEvent(view=view, text='HelloWorld'))
            elif checkable:
                view_desc = checkbox_frame.replace('@', view_local_id).replace('#', view_text)
                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" text='&'", "")
                # available_actions.append(TouchEvent(view=view))
            elif clickable:  # or long_clickable
                if merge_buttons:
                    # below is to merge buttons, led to bugs
                    clickable_ancestor_id = self._get_ancestor_id(view=view, key='clickable')
                    if not clickable_ancestor_id:
                        clickable_ancestor_id = self._get_ancestor_id(view=view, key='checkable')
                    clickable_children_ids = self._extract_all_children(id=clickable_ancestor_id)
                    if view_id not in clickable_children_ids:
                        clickable_children_ids.append(view_id)
                    view_text, content_description = self._merge_text(clickable_children_ids)
                    checked = self._get_children_checked(clickable_children_ids)
                    # end of merging buttons
                view_desc = btn_frame.replace('@', view_local_id).replace('#', view_text)
                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" text='&'", "")
                # available_actions.append(TouchEvent(view=view))
                if merge_buttons:
                    for clickable_child in clickable_children_ids:
                        if clickable_child in enabled_view_ids and clickable_child != view_id:
                            removed_view_ids.append(clickable_child)
            elif scrollable:
                # print(view_id, 'continued')
                view_desc = scroll_frame.replace('@', view_local_id)
                # available_actions.append(ScrollEvent(view=view, direction='DOWN'))
                # available_actions.append(ScrollEvent(view=view, direction='UP'))
            else:
                view_desc = text_frame.replace('@', view_local_id).replace('#', view_text)
                if content_description:
                    view_desc = view_desc.replace('&', content_description)
                else:
                    view_desc = view_desc.replace(" text='&'", "")
                # available_actions.append(TouchEvent(view=view))

            allowed_actions = ['touch']
            special_attrs = []
            if editable:
                allowed_actions.append('set_text')
            if checkable:
                allowed_actions.extend(['select', 'unselect'])
                allowed_actions.remove('touch')
            if scrollable:
                allowed_actions.extend(['scroll up', 'scroll down'])
                allowed_actions.remove('touch')
            if long_clickable:
                allowed_actions.append('long_touch')
            if checked or selected:
                special_attrs.append('selected')
            view['allowed_actions'] = allowed_actions
            view['special_attrs'] = special_attrs
            view['local_id'] = view_local_id
            if len(special_attrs) > 0:
                special_attrs = ','.join(special_attrs)
                view_desc = view_desc.replace("attr=null", f"attr={special_attrs}")
            else:
                view_desc = view_desc.replace(" attr=null", "")
            view_desc = view_desc.replace("bounds=null", f"bound_box={view_bounds}")
            view_descs.append(view_desc)
            view['desc'] = view_desc.replace(f' id={view_local_id}', '').replace(f' attr={special_attrs}', '')
            indexed_views.append(view)

        # prefix = 'The current state has the following UI elements: \n' #views and corresponding actions, with action id in parentheses:\n '
        state_desc = '\n'.join(view_descs)
        activity = self.foreground_activity.split('/')[-1]
        # print(views_without_id)
        return state_desc, activity, indexed_views

    def _get_self_ancestors_property(self, view, key, default=None):
        all_views = [view] + [self.views[i] for i in self.get_all_ancestors(view)]
        for v in all_views:
            value = self.__safe_dict_get(v, key)
            if value:
                return value
        return default
    


    def _merge_text(self, children_ids):
        texts, content_descriptions = [], []
        for childid in children_ids:
            if not self.__safe_dict_get(self.views[childid], 'visible') or \
                self.__safe_dict_get(self.views[childid], 'resource_id') in \
               ['android:id/navigationBarBackground',
                'android:id/statusBarBackground']:
                # if the successor is not visible, then ignore it!get
                continue          

            text = self.__safe_dict_get(self.views[childid], 'text', default='')
            if len(text) > 50:
                text = text[:50]

            if text != '':
                # text = text + '  {'+ str(childid)+ '}'
                texts.append(text)

            content_description = self.__safe_dict_get(self.views[childid], 'content_description', default='')
            if len(content_description) > 50:
                content_description = content_description[:50]

            if content_description != '':
                content_descriptions.append(content_description)

        merged_text = '<br>'.join(texts) if len(texts) > 0 else ''
        merged_desc = '<br>'.join(content_descriptions) if len(content_descriptions) > 0 else ''
        return merged_text, merged_desc
    
        
    #we define a method to ask any prompt to llama
    def ask_llama(self,view, maxl=200, temp=0.7, full_view_representation=None):
        rappresentation = ""
        for line in full_view_representation.split('\n'):
            type = line.split('<')[1].split(' ')[0]
            line = line.split('>')[1].split('<')[0]
            if len(line) > 0:
                rappresentation += f"<{type}>" + line + "</" + type + ">\n"
        full_view_representation = rappresentation
        # Generate a brief description of the current element based on available attributes
        view_text = self.__safe_dict_get(view, 'text', default='').strip()
        content_description = self.__safe_dict_get(view, 'content_description', default='').strip()
        view_class = self.__safe_dict_get(view, 'class', default='').split('.')[-1]

        # Build a readable summary for the current view
        view_details = (
            f"Current Element:\n"
            f"  - Class: {view_class}\n"
            f"  - Text: {view_text if view_text else 'N/A'}\n"
            f"  - Description: {content_description if content_description else 'N/A'}\n"
        )

        # Construct a structured prompt that incorporates both the global UI representation 
        # and the details of the active view
        prompt = (
                f"<s>[INST] <<SYS>>"
                "You are an assistant that generates only a realistic, context-appropriate text input for a UI element. "
                "Respond with a single word of plain text, no explanations, no formatting, no extra symbols."
                "<</SYS>>\n\n"
                "### Few-Shot Examples:\n\n"
                "Example 1:\n"
                "[UI Representation]\n"
                "<input id=0 text='Email address' bound_box=10,50,310,100>Enter email</input>\n"
                "Output:\n"
                "john.doe@example.com\n\n"
                "Example 2:\n"
                "[UI Representation]\n"
                "<input id=1 text='Search bar' bound_box=20,100,300,150>Search products...</input>\n"
                "<button id=2 text='Search' bound_box=320,100,620,150></button>\n"
                "Output:\n"
                "wireless headphones\n\n"
                "UI Snapshot:\n"
                f"{full_view_representation}\n\n"
                "Focused Element Details:\n"
                f"{view_details}\n\n"
                "Now, based on the UI Snapshot and Focused Element Details, provide only the appropriate text for the focused element."
                "[/INST]\n"
                "OUTPUT: "
            )

        
        print("Prompt sent to Llama:")
        print(prompt)
        
        response = ask_llama_local(prompt=prompt, temp=temp)
        response = response.split('OUTPUT:')[1].strip().replace('\n', ' ')
        if len(response) > 15:
            response = response[:15]
        if response:
            print("Response received:")
            print(response)
            print("Response text:")
            print(response.strip())
        else:
            print("Error: No response received from Llama.")
            return None
        # Decode and return the response
        return response.strip()
    
    def ask_choice_llama(self,view, maxl=200, temp=0.7, full_view_representation=None,possible_actions=None):
        rappresentation = ""
        for line in full_view_representation.split('\n'):
            type = line.split('<')[1].split(' ')[0]
            line = line.split('>')[1].split('<')[0]
            if len(line) > 0:
                rappresentation += f"<{type}>" + line + "</" + type + ">\n"
        full_view_representation = rappresentation
        # Generate a brief description of the current element based on available attributes
        view_text = self.__safe_dict_get(view, 'text', default='').strip()
        content_description = self.__safe_dict_get(view, 'content_description', default='').strip()
        view_class = self.__safe_dict_get(view, 'class', default='').split('.')[-1]

        # Build a readable summary for the current view
        view_details = (
            f"Current Element:\n"
            f"  - Class: {view_class}\n"
            f"  - Text: {view_text if view_text else 'N/A'}\n"
            f"  - Description: {content_description if content_description else 'N/A'}\n"
        )
        
        # If no possible actions are provided, return None
        view_possible_actions = None
        if possible_actions:
            # Enumerate possible actions with an index
            view_possible_actions = ""
            for idx, action in enumerate(possible_actions):
                view_possible_actions += f"{idx}: {action}\n"

        # Construct a structured prompt that incorporates both the global UI representation
        # and the details of the active view
        prompt = (
                f"<s>[INST] <<SYS>>"
                "You are an assistant that generates only a realistic, context-appropriate text input for a UI element. "
                "Respond with a single action to take, no explanations, no formatting, no extra symbols."
                "<</SYS>>\n\n"
                "### Few-Shot Examples:\n\n"
                "Example 1:\n"
                "[UI Representation]\n"
                "<input id=0 text='Email address' bound_box=10,50,310,100>Enter email</input>\n"
                "Output:\n"
                "1"
                "Example 2:\n"
                "[UI Representation]\n"
                "<input id=1 text='Search bar' bound_box=20,100,300,150>Search products...</input>\n"
                "<button id=2 text='Search' bound_box=320,100,620,150></button>\n"
                "Output:\n"
                "0"
                "UI Snapshot:\n"
                f"{full_view_representation}\n\n"
                "Focused Element Details:\n"
                f"{view_details}\n\n"
                "Possible Actions:\n"
                f"{view_possible_actions}\n"
                "History of Actions:\n"
                f"{' '.join(f'- {action}' for action in last_actions)}\n"
                "Now, based on the UI Snapshot, Focused Element Details, possible actions and History of Actions, provide only the appropriate index action."
                "[/INST]\n"
                "OUTPUT: "
            )

        
        print("Prompt sent to Llama:")
        print(prompt)
        
        response = ask_llama_local(prompt=prompt, temp=temp)
        response = response.split('OUTPUT:')[1].strip().replace('\n', ' ')
        if len(response) > 15:
            response = response[:15]
        if response:
            print("Response received:")
            print(response)
            print("Response text:")
            print(response.strip())
        else:
            print("Error: No response received from Llama.")
            return None
        # Decode and return the response
        if response.isdigit():
            response = int(response)
            if 0 <= response < len(possible_actions):
                return possible_actions[response]
            else:
                print(f"Error: Response index {response} out of range for possible actions.")
                return None
        else:
            print(f"Error: Response '{response}' is not a valid index.")

def ask_llama_API(self,view, maxl=200, temp=0.7, full_view_representation=None):

        # Generate a brief description of the current element based on available attributes
        view_text = self.__safe_dict_get(view, 'text', default='').strip()
        content_description = self.__safe_dict_get(view, 'content_description', default='').strip()
        view_class = self.__safe_dict_get(view, 'class', default='').split('.')[-1]
        allowed_actions = view.get('allowed_actions', [])

        # Build a readable summary for the current view
        view_details = (
            f"Current Element:\n"
            f"  - Class: {view_class}\n"
            f"  - Text: {view_text if view_text else 'N/A'}\n"
            f"  - Description: {content_description if content_description else 'N/A'}\n"
            f"  - Allowed Actions: {', '.join(allowed_actions) if allowed_actions else 'N/A'}\n"
        )

        # Construct a structured prompt that incorporates both the global UI representation 
        # and the details of the active view
        prompt = (
            f"""<s>[INST] <<SYS>>
        You are a helpful assistant designed to generate context-aware text inputs for UI elements. 
        Analyze the interface structure, element properties, and existing content to suggest realistic values.
        <</SYS>>

        ### Task Description:
        1. Given the current UI state and a focus element (marked as editable):
        2. Suggest a text input that matches the element's context.
        3. The output MUST be few words, without any additional explanations or formatting or simbols.
        4. The output MUST be a single line of text.
        5. The output MUST be different from previus texts.
        5. Follow these guidelines:
        - Use actual data formats (emails, names, numbers) when detectable
        - Mirror the style of existing content when applicable
        - Prioritize content descriptions over placeholder texts
        - Keep inputs minimal but meaningful

        ### Few-Shot Examples:

        Example 1:
        [UI Representation]
        <input id=0 text='Email address' bound_box=10,50,310,100>Enter email</input>

        Output:
        john.doe@example.com

        Example 2: 
        [UI Representation]
        <input id=1 text='Search bar' bound_box=20,100,300,150>Search products...</input>
        <button id=2 text='Search' bound_box=320,100,620,150></button>

        Output:
        wireless headphones

        ### Current Interface:
        {full_view_representation}

        ### Focus Element Details:
        {view_details}

        ### Instruction:
        Based on the above context, suggest appropriate text for the focus element. 
        Provide only the raw text input without explanations, simbols or formatting, JUST TEXT. [/INST]
        
        OUTPUT:
        """
        )
        
        print("Prompt sent to Llama:")
        print(prompt)

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            })
        )

        if response.ok:
            print("Response received:")
            print(response.json())  # Print the JSON response
            print("Response text:")
            print(response.json().get('choices')[0].get('message').get('content').strip())
            
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

        # Decode and return the response
        return response.json().get('choices')[0].get('message').get('content').strip()




